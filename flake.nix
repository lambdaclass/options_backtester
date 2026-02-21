{
  description = "Options backtester dev environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
    in {
      devShells = forAllSystems (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          python = pkgs.python312;
          pythonPkgs = python.pkgs;
        in {
          default = pkgs.mkShell {
            packages = [
              (python.withPackages (ps: [
                # Runtime
                ps.pandas
                ps.numpy
                ps.altair
                ps.pyprind
                ps.seaborn
                ps.matplotlib
                ps.pyarrow

                # Testing
                ps.pytest

                # Dev tools
                ps.yapf
                ps.flake8

                # Data fetching (optional, for data/ scripts)
                ps.yfinance
              ]))
            ];
          };
        });
    };
}
