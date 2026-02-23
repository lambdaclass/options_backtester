{
  description = "Options backtester dev environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = { self, nixpkgs, rust-overlay }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
    in {
      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [ rust-overlay.overlays.default ];
          };
          python = pkgs.python312;
          pythonPkgs = python.pkgs;
          rustToolchain = pkgs.rust-bin.stable.latest.default.override {
            extensions = [ "rust-src" "rust-analyzer" ];
          };
        in {
          default = pkgs.mkShell {
            packages = [
              # Rust
              rustToolchain
              pkgs.maturin
              pkgs.cargo-nextest

              (python.withPackages (ps: [
                # Runtime
                ps.pandas
                ps.numpy
                ps.altair
                ps.pyprind
                ps.seaborn
                ps.matplotlib
                ps.pyarrow

                # Notebooks
                ps.jupyter
                ps.nbconvert
                ps.ipykernel

                # Testing
                ps.pytest
                ps.mypy

                # Dev tools
                ps.yapf
                ps.flake8

                # Data fetching (optional, for data/ scripts)
                ps.yfinance
              ]))
            ];

            shellHook = ''
              export PYO3_PYTHON=${python}/bin/python
            '';
          };
        });
    };
}
