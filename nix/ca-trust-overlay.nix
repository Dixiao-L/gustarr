# eMudhra CA compromise: https://emudhra.we-have-a-problem.com/
final: prev:
let
  distrustedRoots = [ "Comodo AAA Services root" "emSign ECC Root CA - G3" ];
in
{
  cacert = (prev.cacert.override (old: {
    blacklist = (old.blacklist or [ ]) ++ distrustedRoots;
  })).overrideAttrs (old: {
    # Newer buildcatrust preserves blocked roots as TRUSTED CERTIFICATE
    # records with rejection flags. Remove their DER records as well for
    # clients that do not implement OpenSSL auxiliary trust metadata.
    installPhase = old.installPhase + ''
      ${final.python3}/bin/python3 ${./prune-ca-bundles.py} "$out" "$unbundled" "$hashed"
    '';
  });
  uv = prev.uv.overrideAttrs (old: rec {
    version = "0.13.4";
    src = final.fetchFromGitHub {
      owner = "astral-sh";
      repo = "uv";
      tag = version;
      hash = "sha256-US3SjwhWE3v4/i46YuBon2GA6PupC4Z0NRUqcV9CPLM=";
    };
    cargoDeps = final.rustPlatform.fetchCargoVendor {
      inherit src;
      name = "uv-${version}-vendor";
      hash = "sha256-9L47CTlkOVYfGuzE4e7e2FYAI192pAsipzzScF7B1j4=";
    };
  });
  pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
    (pyFinal: pyPrev: {
      certifi = pyPrev.certifi.overridePythonAttrs (old: {
        version = "2026.10.19";
        pyproject = null;
        format = "wheel";
        src = final.fetchurl {
          url = "https://files.pythonhosted.org/packages/py3/c/certifi/certifi-2026.10.19-py3-none-any.whl";
          sha256 = "42d35340cc5b9ba78e818ddbff0c5b1c2b779de448a340fd1b989f5fa8670e1b";
        };
        nativeBuildInputs = [ ];
        nativeCheckInputs = [ ];
        patches = [ ];
        postPatch = "";
        # wheelUnpackHook keeps the wheel in dist/. Apply Nix's existing
        # environment-variable patch after installer has unpacked it.
        postInstall = (old.postInstall or "") + ''
          pushd "$out/${pyFinal.python.sitePackages}"
          ${final.lib.concatMapStringsSep "\n" (patch: "patch -p1 < ${patch}") (old.patches or [ ])}
          rm certifi/cacert.pem
          ln -s ${final.cacert}/etc/ssl/certs/ca-bundle.crt certifi/cacert.pem
          popd
        '';
        doCheck = false;
        pythonImportsCheck = [ "certifi" ];
      });
      # pip vendors certifi; changing the standalone package is insufficient.
      pip = pyFinal.buildPythonPackage {
        pname = "pip";
        version = "26.2.2";
        format = "wheel";
        src = final.fetchurl {
          url = "https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl";
          sha256 = "fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70";
        };
        doCheck = false;
        pythonImportsCheck = [ "pip" ];
        meta = pyPrev.pip.meta;
      };
    })
  ];
}
