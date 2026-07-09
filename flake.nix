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
          config.allowUnfreePredicate = pkg:
            let n = lib.getName pkg; in
            builtins.elem n [ "torch-bin" "triton-bin" ]
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
            };
          };
        in
        python.pkgs.buildPythonApplication {
          pname = "gustarr" + lib.optionalString withMl "-ml";
          version = "0.1.0";
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
