# csv to json conversion module for benchmarks

## Application logic (draft)

The principle is to pre-generate the conversion of the benchmarks csv to json so that the client only has to consume the data.

### Client side

On the client side, we only fetch converted json data, hosted on the static github pages. 

Note:
- In the case of a download failure, default json data, with a speed of 0, is generated to avoid any anomalies and display the problem.
- However, a real data json file, the last one generated, should always be present on the site and accessible through the fetch.

### Node module for converting csv to json

First it is necessary to copy the new benchmarks csv file into the `input` directory of the module.
The module reads and parses the `csv` file and generates a `benchmark.json` file in the `/javascripts` directory of the site, before it is generated into a static site and deployed.

If an error occurs during this process, no json file will be generated, only error logs.
The default json file, or the previously generated file, will be used. In the worst case, 0 data will be displayed by the client.

## Point of attention to solve

Beware, the integration of the module in the `/docs` directory generates some side effects.
Some `node_modules` files potentially conflict with the site build (`The following pages exist in the docs directory, but are not included in the "nav" configuration:`) and pollute the static site `sitemap.xml`.

A solution is to be found.

## To do

- Trouver une solution pour ne pas polluer le build et le sitemap.xml
- Intégrer la conversion dans le chaine de build et de déploiement
- Réaliser des tests unitaire de conversion de csv en json
- Bonus: intégration de typescript avec Vite.

## Node version used

`lts/hydrogen`

