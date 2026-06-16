# Vendored dependencies

This directory contains a copy of `bolt_pipeliner` so the project can
run end-to-end without `pip install bolt_pipeliner`.

**Do not edit files under `bolt_pipeliner/` here.** They are overwritten
on the next `bolt init --refresh-vendor` (or by re-running `bolt init`
in an empty directory). Patch the upstream package instead.
