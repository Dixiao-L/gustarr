# NixOS module: gustarr as systemd timers + optional web UI.
#
# Design: no daemon does the thinking. `gustarr-nightly` (sync→…→rank)
# and `gustarr-weekly` (…→apply) are oneshot services on timers; the only
# long-running process is the approval web UI. State (SQLite store, HF
# model cache) lives in /var/lib/gustarr. Secrets arrive exclusively via
# EnvironmentFile= (agenix-friendly) — the generated TOML config contains
# only `env:VAR` references, never key material.
flake: { config, lib, pkgs, ... }:

let
  cfg = config.services.gustarr;
  settingsFormat = pkgs.formats.toml { };
  configFile = settingsFormat.generate "gustarr.toml" cfg.settings;

  # CUDA userspace comes bundled in torch-bin's wheel, but libcuda.so
  # (the driver stub) must come from the host driver. NixOS exposes it
  # at /run/opengl-driver/lib.
  gpuEnv = {
    LD_LIBRARY_PATH = "/run/opengl-driver/lib";
    HF_HOME = "/var/lib/gustarr/hf";
  };

  commonService = {
    User = "gustarr";
    Group = "gustarr";
    StateDirectory = "gustarr";
    EnvironmentFile = cfg.environmentFiles;
    # Hardening: these jobs only need the store dir, the network, and
    # (for embed) the GPU device nodes.
    ProtectSystem = "strict";
    ProtectHome = true;
    PrivateTmp = true;
    NoNewPrivileges = true;
    SupplementaryGroups = lib.optionals cfg.gpu [ "video" ];
  };
in
{
  options.services.gustarr = {
    enable = lib.mkEnableOption "gustarr media recommendation subsystem";

    package = lib.mkOption {
      type = lib.types.package;
      default =
        if cfg.gpu
        then flake.packages.${pkgs.system}.ml
        else flake.packages.${pkgs.system}.default;
      defaultText = "gustarr from the flake (ml variant when gpu = true)";
      description = "gustarr package to run.";
    };

    gpu = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Use the CUDA ml variant and grant GPU device access to embed/train.";
    };

    settings = lib.mkOption {
      type = settingsFormat.type;
      default = { };
      description = ''
        gustarr.toml contents. Use "env:VAR" for every secret and list the
        env file carrying VAR in environmentFiles.
      '';
    };

    environmentFiles = lib.mkOption {
      type = lib.types.listOf lib.types.path;
      default = [ ];
      description = "EnvironmentFile= entries (agenix .path values) providing the env:VAR secrets.";
    };

    extraEnvironment = lib.mkOption {
      type = lib.types.attrsOf lib.types.str;
      default = { };
      description = "Extra env vars for all units (e.g. HF_ENDPOINT for a HuggingFace mirror).";
    };

    nightly = {
      enable = lib.mkOption { type = lib.types.bool; default = true; };
      onCalendar = lib.mkOption {
        type = lib.types.str;
        default = "*-*-* 04:30:00";
        description = "sync → enrich → candidates → embed → train → rank.";
      };
    };

    weekly = {
      enable = lib.mkOption { type = lib.types.bool; default = true; };
      onCalendar = lib.mkOption {
        type = lib.types.str;
        default = "Sat *-*-* 09:00:00";
        description = "nightly stages + apply (actuates the *arrs within caps).";
      };
    };

    web = {
      enable = lib.mkOption { type = lib.types.bool; default = true; };
    };
  };

  config = lib.mkIf cfg.enable {
    users.users.gustarr = {
      isSystemUser = true;
      group = "gustarr";
    };
    users.groups.gustarr = { };

    systemd.services.gustarr-nightly = lib.mkIf cfg.nightly.enable {
      description = "gustarr nightly pipeline (learn + rank)";
      after = [ "network-online.target" ];
      wants = [ "network-online.target" ];
      environment = gpuEnv // cfg.extraEnvironment;
      serviceConfig = commonService // {
        Type = "oneshot";
        ExecStart = "${lib.getExe cfg.package} --config ${configFile} run nightly";
        # embed can legitimately take a while on first run (model download
        # + full-library embedding); don't let systemd kill a working job.
        TimeoutStartSec = "2h";
      };
    };
    systemd.timers.gustarr-nightly = lib.mkIf cfg.nightly.enable {
      wantedBy = [ "timers.target" ];
      timerConfig = {
        OnCalendar = cfg.nightly.onCalendar;
        RandomizedDelaySec = 600;
        Persistent = true;
      };
    };

    systemd.services.gustarr-weekly = lib.mkIf cfg.weekly.enable {
      description = "gustarr weekly pipeline (learn + rank + apply)";
      after = [ "network-online.target" ];
      wants = [ "network-online.target" ];
      environment = gpuEnv // cfg.extraEnvironment;
      serviceConfig = commonService // {
        Type = "oneshot";
        ExecStart = "${lib.getExe cfg.package} --config ${configFile} run weekly";
        TimeoutStartSec = "2h";
      };
    };
    systemd.timers.gustarr-weekly = lib.mkIf cfg.weekly.enable {
      wantedBy = [ "timers.target" ];
      timerConfig = {
        OnCalendar = cfg.weekly.onCalendar;
        RandomizedDelaySec = 600;
        Persistent = true;
      };
    };

    systemd.services.gustarr-web = lib.mkIf cfg.web.enable {
      description = "gustarr approval web UI";
      wantedBy = [ "multi-user.target" ];
      after = [ "network-online.target" ];
      wants = [ "network-online.target" ];
      environment = cfg.extraEnvironment;
      serviceConfig = commonService // {
        ExecStart = "${lib.getExe cfg.package} --config ${configFile} web";
        Restart = "on-failure";
        RestartSec = 5;
      };
    };
  };
}
