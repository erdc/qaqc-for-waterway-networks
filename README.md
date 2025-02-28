# Quality Control for Waterway Networks

Developed by:  
Connor Hicks  
Junior Software Engineer, ARA

Magdalena Asborno, PhD  
Senior Consultant, ARA  
Research Civil Engineer, USACE-ERDC

## Cite this work

If you use this work, please add this citation:

Asborno, M., C. Hicks, and K. N. Mitchell. 2025. Quality Control for Waterway Network ERDC/CHL CHETN-IX-##. Vicksburg, MS: US Army Engineer Research and Development Center, Coastal and Hydraulics Laboratory. DOI forthcoming

## About

The "Quality Control for Waterway Networks" Process is an automated process to update the U.S. Army Corps of Engineer’s (USACE) Engineer Research and Development Center (ERDC) Waterway Network.

After a user introduces desired changes to a line layer representing the waterways, the process creates a fully connected network, and controls topology quality. This process also updates waterway depths and geometries based on the most recent version of the USACE National Channel Framework (NCF) and performs spatial joins of network nodes with other various sources of data. Users currently have 2 options of running this process as a Python script: through the QGIS Toolbox interface, or through a standalone terminal.

## Toolbox Script

### Prerequisites

1. Install QGIS

### How to run

1. Download and unzip the latest release [here.](https://github.com/erdc/waterway-network/releases/latest)
2. Open a QGIS project with the layer you wish to update, or add the desired layer.
3. In the top menu, click `Layer` > `Add Layer` > `Add ArcGIS REST Server Layer...`
4. Click `New`. Enter a name for this server connection.
5. The URL should point to the National Channel Framework API. Usually is [https://services7.arcgis.com/n1YM8pTrFmm7L4hs/ArcGIS/rest/services/National_Channel_Framework/FeatureServer](https://services7.arcgis.com/n1YM8pTrFmm7L4hs/ArcGIS/rest/services/National_Channel_Framework/FeatureServer).
6. Click `OK`. Click `Connect`.
7. Select the ChannelReach polygon layer. Click `Add` or `Add with Filter`.
8. (Optional) Download your State layer. The URL to the download usually is [https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_5m.zip](https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_5m.zip).
9. (Optional) Extract the .zip file, and drag the downloaded State shapefile into your QGIS project.
10. (Optional) Download your County layer. Typically, this can be found here: [https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_5m.zip](https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_5m.zip).
11. (Optional) Extract the .zip file, and drag the downloaded County shapefile into your QGIS project.
12. Navigate to your Processing Toolbox window. If you do not see this window, in the top menu, click `Processing` and ensure that `Toolbox` is enabled.
13. Click the 2nd icon in the top row of the Processing Toolbox window > `Add Script to Toolbox...`

![Screenshot 2024-11-06 142555](https://github.com/user-attachments/assets/e22fd81c-5442-4181-a781-51129e7f53d2)

15. Navigate to your downloaded Toolbox Script file. This adds a shortcut to your QGIS installation to the Toolbox Script file for future use.
16. Scroll down the Processing Toolbox window until you see `Scripts`. Click `Scripts` > `Quality Control` > `Quality Control for Waterway Networks`.

![Screenshot 2024-11-06 142632](https://github.com/user-attachments/assets/c5de932d-fd25-4968-9458-14c06e3280a7)

17. Enter the desired parameters. Parameters in the **INPUTS** and **SETTINGS** sections are required, unless stated otherwise. The help window to the right provides additional insights behind each parameter. Below is an example screenshot of the QGIS Toolbox menu with completed parameters.
18. Click `Run`.

![Screenshot 2024-10-16 164351](https://github.com/user-attachments/assets/13625f0b-6a01-4e85-83e3-18d2e64058ae)

## Versioning

This project follows [Semantic Versioning](https://semver.org/) (SemVer) principles to ensure consistent and predictable version numbering. The version number is automatically incremented on every commit or merge to the `main` branch based on commit message conventions.

### Version Increment Rules

The version number (MAJOR.MINOR.PATCH) is incremented according to the following rules:

#### MAJOR Version Bump (X.0.0)

A major version increment occurs when a commit message contains any of:

- `BREAKING CHANGE`
- `!:`
- `major:`

Use these when making incompatible API changes.

#### MINOR Version Bump (0.X.0)

A minor version increment occurs when a commit message contains either:

- `feat:`
- `minor:`

Use these when adding functionality in a backwards compatible manner.

#### PATCH Version Bump (0.0.X)

A patch version increment occurs automatically for any commit that doesn't contain the above keywords.

Use this for backwards compatible bug fixes and minor changes.

### Commit Message Examples

```
major: restructure transformation pipeline
# This will trigger a MAJOR version bump

feat: add new coordinate transformation option
# This will trigger a MINOR version bump

style: improve code formatting
# This will trigger a PATCH version bump
```
