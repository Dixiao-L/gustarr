{
  description = "gustarr — single-user media taste learning + Servarr automation";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      lib = nixpkgs.lib;
      systems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" ];
      forAll = lib.genAttrs systems;

      # torch-bin (prebuilt wheel, bundled CUDA runtime on Linux) instead of
      # source-built torch: a source CUDA torch build takes hours and needs
      # the full cudaPackages toolchain; the wheel just works against any
      # >=525 driver via /run/opengl-driver/lib. The override maps every
      # consumer (sentence-transformers → transformers → ...) onto the same
      # torch-bin instance.
      mkPkgs = system:
        import nixpkgs {
          inherit system;
          # torch-bin 2.12 wheels are built against CUDA 13 (cuda-bindings
          # >=13.0.3); nixpkgs still defaults cudaPackages to 12.9 and
          # refuses to evaluate. Any >=525 driver runs CUDA-13 userspace,
          # so this is a metadata unblock, not a compatibility gamble.
          overlays = [ (final: _prev: { cudaPackages = final.cudaPackages_13; }) ];
          config.allowUnfreePredicate = pkg:
            let n = lib.getName pkg; in
            # torch-bin evaluates under pname "torch" (nixpkgs shares the
            # name across source/bin variants); its license is
            # bsd3+issl+unfreeRedistributable purely from the bundled
            # NVIDIA userspace.
            builtins.elem n [
              "torch" "torch-bin" "triton" "triton-bin"
              "cudnn" "nccl" "cutensor" "cusparselt"
            ]
            || lib.hasPrefix "cuda" n
            || lib.hasPrefix "libcu" n
            || lib.hasPrefix "libnv" n
            || lib.hasPrefix "nvidia" n;
        };

      mkGustarr = pkgs: withMl:
        let
          python = pkgs.python312.override {
            packageOverrides = _self: super: {
              torch = super.torch-bin;
              # Anything downstream that reaches for the source torchvision
              # would inherit attrs torch-bin doesn't expose (cudaSupport)
              # and trigger an hours-long source build besides.
              torchvision = super.torchvision-bin;
              # sentence-transformers' nativeCheckInputs pull in ALL its
              # optional-dependency sets (transformers audio extras →
              # torchaudio) just to run its own test suite. nixpkgs'
              # torchaudio-bin is still the cu12 wheel (2.11, wants
              # libcudart.so.12) and cannot patchelf against the cu13 stack
              # torch-bin 2.12 brought in — and we don't need to re-run
              # upstream's tests to package a consumer. pythonImportsCheck
              # still smoke-tests the import.
              torchaudio = super.torchaudio-bin;
              sentence-transformers = super.sentence-transformers.overridePythonAttrs (_old: {
                doCheck = false;
                nativeCheckInputs = [ ];
              });
            };
          };
        in
        python.pkgs.buildPythonApplication {
          pname = "gustarr" + lib.optionalString withMl "-ml";
          version = "0.4.0";
          pyproject = true;
          src = lib.cleanSource ./.;

          build-system = [ python.pkgs.hatchling ];

          dependencies = with python.pkgs; [
            click
            httpx
            numpy
            fastapi
            uvicorn
          ] ++ lib.optionals withMl [
            torch
            sentence-transformers
          ];

          # The suite is fully offline by design (httpx.MockTransport);
          # anything that would touch the network is a bug the sandbox
          # rightly catches.
          nativeCheckInputs = [ python.pkgs.pytestCheckHook ];

          meta = {
            description = "Learns one user's media taste and drives Sonarr/Radarr/Lidarr";
            mainProgram = "gustarr";
            license = lib.licenses.mit;
          };
        };
    in
    {
      packages = forAll (system:
        let pkgs = mkPkgs system; in
        {
          default = mkGustarr pkgs false;
        }
        # The ML variant (GPU embedding) is Linux-only: that's where it
        # deploys, and it keeps darwin dev shells light.
        // lib.optionalAttrs (lib.hasSuffix "linux" system) {
          ml = mkGustarr pkgs true;
        });

      nixosModules.default = import ./nix/module.nix self;

      devShells = forAll (system:
        let pkgs = mkPkgs system; in
        {
          default = pkgs.mkShell {
            packages = [ pkgs.uv pkgs.python312 ];
          };
        });
    };
}
