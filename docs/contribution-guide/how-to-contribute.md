# Contributing guidelines

Our goal is to **make kernl development smooth**, **stable** and **open**.
We can't do this without our user community!
Please read this document before contributing!

## Code of Conduct

Kernl has adopted the **Contributor Covenant** as its **[Code of Conduct](code-of-conduct.md)**, and we expect project participants to adhere to it. 
Please read the **[full text](code-of-conduct.md)** so that you can understand what actions will and will not be tolerated.

## Open Development

Kernl uses **[GitHub](https://github.com/ELS-RD/kernl)** as a source of truth. The core team works directly on it. All changes are public.

## Writing and formatting text

In a global way, texts in Issues, PRs and commits should be written **in English**, **with care**, **consistency**, and **clarity**.
Use (**[Markdown](https://www.markdownguide.org/basic-syntax/)**) to structure your texts, and to make them more readable.

## Development Process

### Ask a question / share your ideas or new features

If you have a question, an idea for a new feature, or an optimization, you can **use the dedicated [discussion forum](https://github.com/ELS-RD/kernl/discussions)**.

### Issues

!!! warning "Before submitting an Issue, check the **[issue tracker](https://github.com/ELS-RD/kernl/issues)** to see if a similar version is already present"

#### Bugs

If you think you've found a bug but don't have the time to send a PR, please:

- First, **make sure the bug has not already been reported** in the **[issue tracker](https://github.com/ELS-RD/kernl/issues)**.
- If you don't find a similar Issue, **[open a new one](todo)**.

Bug reports without a reproduction will be immediately closed.

!!! tip "tips for writing your Issue"

    - [x] Give a concise and explicit title to the problem so that you can quickly understand what it is about.
    - [x] Describe the problem clearly and in detail. Explain what is not working as expected, what you expected to happen and what happened instead. Include as much information as possible, all the steps you need to take to reproduce the bug, etc. If you have ideas on how to fix the problem, feel free to include them in your description.

#### Feature requests

If you would like to **request a new feature** or **an enhancement**, 
but are not planning to open a Pull Request, you can **[open a dedicated issue](todo)**.
You can also use the **[discussion forum](todo)** for feature requests that you would like to discuss.

!!! example "@reviewer: see [feature request template](https://github.com/ELS-RD/kernl/blob/feat/contribution-guide-doc/.github/ISSUE_TEMPLATE/feature.yml)"

#### Proposals

If you intend to **make non-trivial changes to existing implementations**, we recommend that you **file an Issue** with the **[proposal template](todo)**. This allows us to reach agreement on your proposal before you put any effort into it.

!!! example "@reviewer: see [proposal template](https://github.com/ELS-RD/kernl/blob/feat/contribution-guide-doc/.github/ISSUE_TEMPLATE/proposal.yml)"

#### Documentation

If you would like to **add or improve documentation** or **tutorials**,
but are not planning to open a Pull Request, you can **[open a dedicated issue](todo)**.
You can also use the **[discussion forum](todo)** to discuss this topic.

!!! example "@reviewer: see [documentation template](https://github.com/ELS-RD/kernl/blob/feat/contribution-guide-doc/.github/ISSUE_TEMPLATE/documentation.yml)"

## Development

### Branch Organization

Kernl works with a `main` branch. This is the source of truth.

Each **new Pull Request**, for a new feature, bug fix, or other, must be done on **a new branch**.

#### Standardization of branch names

The name of a new branch must respect the following format:
```
<type>(scope)/<subject>
``` 

??? question "Which types and characters are allowed?"

    The types of new branches are inspired by the values of **[Conventional commit](https://www.conventionalcommits.org/en/v1.0.0/)**.
    
    **List of possible types:**

    - `feat`: new functionality, optimization or implementation of a proposal.
    - `fix`: bug fix.
    - `docs`: ajout ou modification de la documentation.
    - `refactor`: a change in the code that does not result in any difference in behavior.
    - `test`: adding tests, refactoring tests. No production code change.
    - `chore`: upgrading dependencies, releasing new versions. Tasks that are regularly done for maintenance purposes.
    - `misc`: anything else that doesn't change production code, yet is not test or chore. e.g. updating GitHub actions workflow.

    **Allowed characters for the subject:** 
        ```regexp
        [a-z_-]
        ```
    !!! example "Examples"
        - feat/debugger
        - feat/backward_layernorm
        - refactor/refactor-kernels
        - docs/contribution-guide-doc

### Installation

!!! warning "IMPORTANT"

    This package requires `pytorch` being installed.  
    Please **install it first**.

```{.shell .copy}
pip install 'git+https://github.com/ELS-RD/kernl' --extra-index-url https://download.pytorch.org/whl/nightly/cu117
```

This project requires `Python` >= 3.9.

### Getting started

```{.python .copy}
import torch
from transformers import AutoModel
from kernl.model_optimization import optimize_model

model = AutoModel.from_pretrained(model_name).eval().cuda()
optimize_model(model)

inputs = ...

with torch.inference_mode(), torch.cuda.amp.autocast():
    outputs = model(**inputs)
```

For end-to-end use cases, you may want to check:

* [XNLI classication with Roberta](https://github.com/ELS-RD/kernl/tree/main/tutorial/bert%20e2e.ipynb)
* [text generation with T5](https://github.com/ELS-RD/kernl/tree/main/tutorial/t5%20e2e.ipynb)

### Test and Benchmark

#### Conventions

- A test function using benchmark features must have a name that starts with `test_benchmark_`
- Benchmark function must have a param called `implementation` when benchmarking the same operation using different
  strategy

#### Run tests and benchmarks

```{.shell .copy}
# tada!
pytest
```

There are over 2K benchmarks, and they take a while to run.

Some rules on how `PyTest` works, in particular for benchmarks:

- add `-k` to filter tests/benchmarks by their name like `pytest -k benchmark` to run only tests with `benchmark`
  in their name
- you can combine expressions in the filter: `pytest -k "benchmark and not bert"` if you want to run all benchmarks
  except those related to BERT
- to group and compare benchmark measures, use `pytest -k benchmark --benchmark-group-by ...`:
    - groupinng by names: `pytest -k benchmark --benchmark-group-by fullfunc`
    - grouping by names of parameters: `pytest -k benchmark --benchmark-group-by param:implementation,param:shape`
        - `param:x`, `x` is the parameter name in `@pytest.mark.parametrize`
    - combining both: `pytest -k benchmark --benchmark-group-by fullfunc,param:implementation`
- add `-s` to see the output of the tests (print, etc.)
- add `-v` to see the verbose output of the tests

!!! warning "WARNING"

    `param:X` will make PyTest crash if `X` is not a parameter of at least one of the function ran.

Some useful commands:

```{.shell .copy}
# only benchmarks
pytest -k benchmark
# no benchmarks
pytest -k "not benchmark"
# only linear layers benchmark, group by shape and if the input is contiguous or not 
pytest test/test_linear_layer.py --benchmark-group-by fullfunc,param:shape,param:contiguous
```

### Create new patterns to replace fx graph nodes

The first step to replace function/module calls in the graph is to create the pattern that will be replaced.
The easiest way to do this is to [convert the model to a fx graph](https://pytorch.org/docs/stable/fx.html), and then
print it with `utils.graph_report` or by printing the code `print(you_graph_module.code)`

Then you can use [replace_pattern](https://pytorch.org/docs/stable/fx.html#torch.fx.replace_pattern) to replace the
pattern in the graph. We have our own version of `replace_pattern` with some enhancements to work with modules, for
example. You can find examples of that in `optimizer` folder.

### Code Formatting

We use `black` / `isort` / `flake8` to format the code. You can run them with:

```{.shell .copy}
make source_code_format
make source_code_check_format
```

!!! example "@reviewer: the question of the organization of the documentation arises"

    Should the installation, get-started, test, etc., sections be included in the contribution quide, at the risk of duplicating efforts?
    
    Anyway, if we move this part to the documentation. It must be removed from the README. 

## Pull Requests

You want to contribute by opening a Pull Request, we appreciate it. We'll do our best to work with you and get the PR reviewed.

> If you're working on your first Pull Request, you can learn how from this free video series from Kent C. Dodds "[How to Contribute to an Open Source Project on GitHub](https://egghead.io/courses/how-to-contribute-to-an-open-source-project-on-github)" and "[About pull requests](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests)".

### Recommendations

- Any new Pull Request must be **created from the `main` branch**.
- The **name** of the new branch must **follow the [standardization](#standardization-of-branch-names)**
- A PR should be kept as **small as possible**. Smaller PRs are much easier to examine and merge.
- Make sure the PR only does **one thing**, otherwise split it up.
- Write a **descriptive title**. It is recommended to **follow this [commit message style](#semantic-commit-messages)**.
- The **description should be clear and structured** to improve readability.
- When relevant, remember to **reference the issue** with `fix #issue_number`.
- Finally, while your PR is in progress, **leave it as a draft**. **Once finalized, make it ready for review**.

!!! success "contribution is more important than following any procedure"

    The maintainers will review your code and point out obvious problems.
    Your contribution is more important than following any procedure, although following these recommendations will certainly save everyone time.

### Breaking Changes

When adding a new Breaking change, we recommend following this template in your Pull Request description:

```{.markdown .copy}
### Breaking change

- **Who does this affect**:
- **How to migrate**:
- **An index to measure the severity and effort required for migration**:
```

### Semantic Commit Messages 

Commit messages must respect the **[Conventional commit](https://www.conventionalcommits.org/en/v1.0.0/)** specification.

The principle is simple, the commit must respect the format:
```
<type>(<scope>): <subject>
```

The types are described in the [specification](https://www.conventionalcommits.org/en/v1.0.0/), the message must be **lower case**.

??? warning "Please pay special attention to the breaking changes"

    A breaking change MUST be indicated by a `!` immediately before the `:`.

    !!! example

        `fix!: fake commit message.` 




