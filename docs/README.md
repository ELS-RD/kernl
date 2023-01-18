# Kernl static documentation site

## Targeting the Material for MkDocs Insiders version

To avoid any breaking change, we target a specific version.

For the preview (`doc/Dockerfile.mkdocstrings`)
```dockerfile
FROM ghcr.io/els-rd/mkdocs-material-insiders:version

# Ex.
FROM ghcr.io/els-rd/mkdocs-material-insiders:4.28.0
```

In the Github Action workflow (`.github/workflows/deploy-static-site.yml`)
```yaml
run: pip install git+https://${{secrets.GH_TOKEN}}@github.com/squidfunk/mkdocs-material-insiders.git@version "mkdocstrings[python]"

# Ex.:
run: pip install git+https://${{secrets.GH_TOKEN}}@github.com/squidfunk/mkdocs-material-insiders.git@9.0.5-insiders-4.28.0 "mkdocstrings[python]"
```

### Version upgrade

- Check the [changelog](https://squidfunk.github.io/mkdocs-material/insiders/changelog/) to verify the current version and take into account the breaking changes if any. 
- Synchronize the fork manually (@pommedeterresautee has the rights).
- Get the version number from [the changelog](https://github.com/ELS-RD/mkdocs-material-insiders/blob/master/CHANGELOG) of the repo.
  - Ex.: `mkdocs-material-9.0.5+insiders-4.28.0` (it must respect this pattern).
- Create a release manually with :
  - As name, the complete version number. Ex. : `mkdocs-material-9.0.5+insiders-4.28.0`
  - As a tag, the insiders version number. Ex. : `4.28.0`
  - This will automatically create the [Docker image in the private registry](https://github.com/orgs/ELS-RD/packages/container/package/mkdocs-material-insiders).
- Change the `doc/Dockerfile.mkdocstrings` file with the insiders version.
  ```dockerfile
  # Ex.
  FROM ghcr.io/els-rd/mkdocs-material-insiders:4.28.0
  ```
- Change the workflow `.github/workflows/deploy-static-site.yml` with the appropriate pattern.
  ```yaml
  # Ex.:
  run: pip install git+https://${{secrets.GH_TOKEN}}@github.com/squidfunk/mkdocs-material-insiders.git@9.0.5-insiders-4.28.0 "mkdocstrings[python]"
  ```

## Preview the static site locally 

The easiest and least intrusive way is to use docker.

```shell
# Building a mkdocs image with the mkdocstrings plugin
docker build -t mkdocstrings -f docs/Dockerfile.mkdocstrings docs

# Previewing the site in watch mode
docker run --rm -it -p 8000:8000 -v ${PWD}:/docs mkdocstrings

# Build the static site
docker run --rm -it -v ${PWD}:/docs mkdocstrings build
```

## Notes for developers

### Use variabilized classes taking into account the dark and light modes

To use the variabilized classes taking into account the dark and light modes (example: `color: var(--md-default-fg-color)`).
You need to add a `data-md-color-scheme="slate" | "default"` metadata to the enclosing elements
