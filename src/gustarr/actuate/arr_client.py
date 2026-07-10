"""Thin Servarr API clients: just enough to add items carrying the
gustarr tag. Read endpoints (tag, profiles, root folders) are cached per
client instance since apply reuses one client for a whole run.
"""

from __future__ import annotations

import json
from typing import Any

from .. import http
from ..config import ArrConfig


class ArrError(Exception):
    """Actuation failure that is not a transport error: bad config
    (unknown quality profile), empty lookup, missing profiles."""


def _is_duplicate_add(exc: http.ApiError) -> bool:
    """True only when the arr's 400 says the item itself already exists
    ("... has already been added" / *ExistsValidator) — NOT any 400 that
    merely mentions "already", e.g. the path collision "Path is already
    configured for an existing movie"."""
    if exc.status != 400:
        return False
    detail = getattr(exc, "detail", "") or str(exc)
    # The body is a JSON list of validation errors, possibly embedded in
    # the exception message and truncated by http.py; parse when we can.
    start = detail.find("[")
    if start >= 0:
        try:
            errors = json.loads(detail[start:])
        except ValueError:
            errors = None
        if isinstance(errors, list):
            return any(
                "already been added" in str(err.get("errorMessage") or "").lower()
                or str(err.get("errorCode") or "").endswith("ExistsValidator")
                for err in errors if isinstance(err, dict))
    # unparseable (truncated) body: fall back to the exact duplicate phrase
    return "already been added" in detail.lower()


class ArrClient:
    api_version = "v3"
    name = "arr"

    def __init__(self, arr: ArrConfig):
        self.cfg = arr
        self.base = arr.url.rstrip("/")
        self._tag_id: int | None = None
        self._profile_id: int | None = None
        self._root: str | None = None

    def _url(self, path: str) -> str:
        return f"{self.base}/api/{self.api_version}/{path}"

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return http.get_json(
            self._url(path), params=params, headers={"X-Api-Key": self.cfg.api_key})

    def _post(self, path: str, body: Any) -> Any:
        return http.post_json(
            self._url(path), json_body=body, headers={"X-Api-Key": self.cfg.api_key})

    def ensure_tag(self) -> int:
        if self._tag_id is None:
            want = self.cfg.tag.lower()
            for tag in self._get("tag"):
                if tag["label"].lower() == want:
                    self._tag_id = tag["id"]
                    break
            else:
                self._tag_id = self._post("tag", {"label": self.cfg.tag})["id"]
        return self._tag_id

    def quality_profile_id(self) -> int:
        if self._profile_id is None:
            profiles = self._get("qualityprofile")
            if not profiles:
                raise ArrError(f"{self.name} at {self.base} has no quality profiles")
            want = self.cfg.quality_profile
            if not want:
                self._profile_id = profiles[0]["id"]
            else:
                by_name = {p["name"].lower(): p["id"] for p in profiles}
                if want.lower() not in by_name:
                    names = ", ".join(p["name"] for p in profiles)
                    raise ArrError(
                        f"{self.name} quality profile {want!r} not found; available: {names}")
                self._profile_id = by_name[want.lower()]
        return self._profile_id

    def root_folder_path(self) -> str:
        if self._root is None:
            folders = self._get("rootfolder")
            if not folders:
                raise ArrError(f"{self.name} at {self.base} has no root folders")
            paths = [f["path"] for f in folders]
            want = self.cfg.root_folder.rstrip("/")
            if not want:
                self._root = paths[0]
            else:
                match = next((p for p in paths if p.rstrip("/") == want), None)
                if match is None:
                    # an explicit setting silently ignored would download
                    # into the wrong place — fail like quality_profile does
                    raise ArrError(
                        f"{self.name} root folder {self.cfg.root_folder!r} not found;"
                        f" available: {', '.join(paths)}")
                self._root = match
        return self._root

    def _post_add(self, path: str, body: dict[str, Any], title: str) -> dict[str, Any]:
        try:
            resp = self._post(path, body)
        except http.ApiError as exc:
            # The *arrs answer 400 "...has already been added" for
            # duplicates — for us that is the desired end state, not an error.
            if _is_duplicate_add(exc):
                return {"existing": True, "title": title}
            raise
        return {"added": True, "title": title, "arr_id": (resp or {}).get("id")}


class RadarrClient(ArrClient):
    name = "radarr"

    def add(self, tmdb_id: int | str) -> dict[str, Any]:
        info = self._get("movie/lookup/tmdb", params={"tmdbId": tmdb_id})
        if isinstance(info, list):
            info = info[0] if info else None
        if not info:
            raise ArrError(f"radarr lookup found nothing for tmdb:{tmdb_id}")
        body = {
            "title": info["title"],
            "tmdbId": info.get("tmdbId", tmdb_id),
            "year": info.get("year"),
            "qualityProfileId": self.quality_profile_id(),
            "rootFolderPath": self.root_folder_path(),
            "monitored": True,
            "tags": [self.ensure_tag()],
            "addOptions": {"searchForMovie": True},
        }
        return self._post_add("movie", body, info["title"])


class SonarrClient(ArrClient):
    name = "sonarr"

    def add(self, tvdb_id: int | str) -> dict[str, Any]:
        results = self._get("series/lookup", params={"term": f"tvdb:{tvdb_id}"})
        if not results:
            raise ArrError(f"sonarr lookup found nothing for tvdb:{tvdb_id}")
        info = results[0]
        body = {
            "title": info["title"],
            "tvdbId": info.get("tvdbId", tvdb_id),
            "year": info.get("year"),
            "qualityProfileId": self.quality_profile_id(),
            "rootFolderPath": self.root_folder_path(),
            "monitored": True,
            "seasonFolder": True,
            "tags": [self.ensure_tag()],
            "addOptions": {"searchForMissingEpisodes": True},
        }
        return self._post_add("series", body, info["title"])


class LidarrClient(ArrClient):
    api_version = "v1"
    name = "lidarr"

    def __init__(self, arr: ArrConfig):
        super().__init__(arr)
        self._metadata_profile_id: int | None = None

    def metadata_profile_id(self) -> int:
        if self._metadata_profile_id is None:
            profiles = self._get("metadataprofile")
            if not profiles:
                raise ArrError(f"lidarr at {self.base} has no metadata profiles")
            self._metadata_profile_id = profiles[0]["id"]
        return self._metadata_profile_id

    def add_artist(self, mbid: str) -> dict[str, Any]:
        # Lidarr's POST /artist validates ArtistName etc., which a bare
        # foreignArtistId can't satisfy — the add body must be the full
        # lookup resource, mirroring the Radarr/Sonarr flows. (Learned in
        # production: every add 400'd with "'Artist Name' must not be
        # empty" while the lenient test mock accepted the bare payload.)
        found = self._get("artist/lookup", params={"term": f"lidarr:{mbid}"}) or []
        if not found:
            raise ArrError(f"lidarr lookup found nothing for mbid {mbid}")
        body = {
            **found[0],
            "qualityProfileId": self.quality_profile_id(),
            "metadataProfileId": self.metadata_profile_id(),
            "rootFolderPath": self.root_folder_path(),
            "monitored": True,
            "tags": [self.ensure_tag()],
            "addOptions": {"monitor": "all", "searchForMissingAlbums": True},
        }
        return self._post_add("artist", body, mbid)
