# CLAUDE.md

This repository is public. Teams can read everything pushed to it: files, commits, branches,
pull requests, issues and review comments, including history that is later rewritten.

## Preliminary and final round tasks stay in the `private/` submodule

The preliminary and final round task packs are secret until the organizers announce them. Teams
must not learn them from this repository, its history, or its GitHub pages. They live only in
`private/`, a submodule of the organizers' private repository
(`matsuolab/geniac_prize_2026_code_private`), marked `update = none` so that teams' clones skip
it. Organizers check it out with `git submodule update --init --checkout private`.

- Never commit them anywhere outside `private/`: task ids, instructions, scenes, variations,
  env code, BDDL files, seeds, initial states, assets, or numbers per task. This covers code,
  tests, fixtures, docs, configs, logs, and examples. Use `taskpacks/original` for anything
  public.
- Do not describe the evaluation tasks: not what they contain, how they differ from
  `taskpacks/original`, or which projects, datasets, forks or simulator variants they are built
  from. Public text says only that the evaluation tasks are private. `private/CLAUDE.md` names
  what this covers.
- Never write their details in commit messages, branch names, PR titles and descriptions, issues,
  review comments, or code comments, in either repository. Say what changed in general terms
  ("Add the final round's LIBERO tasks"), not what the tasks are.
- Code outside `private/` must not import or depend on anything in `private/`. This repository
  builds, tests and evaluates without the submodule.
- If you find round details outside `private/`, stop and tell the organizers instead of
  fixing it in a normal commit; removing them may need a history rewrite.

## Changes that touch both repositories

Merge the change in the private repository first. Then open a pull request here that moves the
`private` submodule to that commit, together with the public side of the change.

## Commit identity

Set `user.name` and `user.email` before committing. Without them git falls back to
`<account>@<host>`, which puts the cluster's account and host names in the public history.
