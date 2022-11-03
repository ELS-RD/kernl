# Kernl static documentation site

## Preview the static site locally 

The easiest and least intrusive way is to use docker.

```shell
# Previewing the site in watch mode
docker run --rm -it -p 8000:8000 -v ${PWD}:/docs squidfunk/mkdocs-material

# Build the static site
docker run --rm -it -v ${PWD}:/docs squidfunk/mkdocs-material build
```

## Notes for developers

### Use variabilized classes taking into account the dark and light modes

To use the variabilized classes taking into account the dark and light modes (example: `color: var(--md-default-fg-color)`).
You need to add a `data-md-color-scheme="slate" | "default"` metadata to the enclosing elements