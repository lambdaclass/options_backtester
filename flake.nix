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
                ps.polars

                # Notebooks
                ps.jupyter
                ps.nbconvert
                ps.ipykernel

                # Testing
                ps.pytest
                ps.hypothesis
                ps.pytest-benchmark
                ps.mypy
                ps.pandas-stubs
                ps.ruff

                # Dev tools
                ps.yapf

                # Data fetching (optional, for data/ scripts)
                ps.yfinance
              ]))
            ];

            shellHook = ''
              export PYO3_PYTHON=${python}/bin/python
              export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1

              # Build Rust extension and symlink for Python import
              if [ -f rust/ob_python/Cargo.toml ]; then
                if [ ! -f rust/target/release/lib_ob_rust.dylib ] && [ ! -f rust/target/release/lib_ob_rust.so ]; then
                  echo "Building Rust extension (first time only)..."
                  cargo build --manifest-path rust/ob_python/Cargo.toml --release 2>&1 | tail -1
                fi
                # Python needs _ob_rust.so, Rust produces lib_ob_rust.dylib/.so
                if [ -f rust/target/release/lib_ob_rust.dylib ] && [ ! -f _ob_rust.so ]; then
                  ln -sf rust/target/release/lib_ob_rust.dylib _ob_rust.so
                elif [ -f rust/target/release/lib_ob_rust.so ] && [ ! -f _ob_rust.so ]; then
                  ln -sf rust/target/release/lib_ob_rust.so _ob_rust.so
                fi
              fi
            '';
          };
        });
    };
}
