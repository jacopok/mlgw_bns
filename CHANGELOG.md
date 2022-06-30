# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `ModesModel` including higher order modes' contributions

## [0.10.1] - 2022-06-30

### Added

- Changelog!
- Some badges in the README:
    - coverage report with [coveralls](https://coveralls.io/)
    - downloads per month

### Changed

- Default model given now starts from 5Hz

### Fixed

- PCA now uses SVD
- Fix TEOB call error, which occurred when the integration time exceeded 1e9M
- Fix `ValidateModel` frequency arrays
- Various fixes to tests

[Unreleased]: https://github.com/jacopok/mlgw_bns/compare/v0.10.1...HEAD
[0.10.1]: https://github.com/jacopok/mlgw_bns/compare/v0.10.0...v0.10.1
