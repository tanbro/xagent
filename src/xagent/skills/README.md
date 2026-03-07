# Skills Directory

This directory contains the skills system for xagent.

## Structure

```
skills/
├── builtin/              # Built-in skills (committed to git)
│   ├── code_reviewer/
│   │   ├── SKILL.md
│   │   └── template.md
│   └── test_generator/
│       ├── SKILL.md
│       └── template.md
└── manager.py           # Skill manager implementation
```

## Built-in Skills

Built-in skills are located in `src/xagent/skills/builtin/` and are committed to the repository.

> **Note:** Currently no built-in skills are included. Add your own skills here to ship with the application.

## User Skills

Users can add custom skills in `.xagent/skills/` (outside the `src/` directory).

User skills:
- Are not committed to git (see `.gitignore`)
- Override built-in skills with the same name
- Can be added without modifying the source code

## Skill Format

Each skill is a directory containing:

- **SKILL.md** (required): Entry point with description, when to use, execution flow
- **template.md** (optional): Prompt template for the skill
- **examples/** (optional): Example files
- **resources/** (optional): Additional resources

See individual skill directories for examples.

## Adding a New Built-in Skill

1. Create a new directory in `src/xagent/skills/builtin/your_skill/`
2. Add `SKILL.md` with the skill description
3. Optionally add `template.md` or other files
4. The skill will be automatically loaded on startup

## Adding a User Skill

1. Create a directory in `.xagent/skills/your_skill/`
2. Add `SKILL.md` and any other files
3. Restart the server or call `POST /api/skills/reload`

## Configuration

### Environment Variables

You can customize the skills directories by setting the `XAGENT_SKILLS_LIBRARY_DIRS` environment variable:

```bash
# Single directory
XAGENT_SKILLS_LIBRARY_DIRS="/path/to/custom/skills"

# Multiple directories (comma-separated)
XAGENT_SKILLS_LIBRARY_DIRS="/path/to/skills1,/path/to/skills2,~/skills"

# With path expansion
XAGENT_SKILLS_LIBRARY_DIRS="~/skills,$HOME/custom_skills,./local_skills"

# The directories are loaded in order, with later ones overriding earlier ones
# for skills with the same name.
```

### Path Expansion Support

- **Home Directory**: `~` is expanded to the user's home directory
- **Environment Variables**: `$VAR` and `${VAR}` syntax are supported
- **Relative Paths**: Converted to absolute paths automatically
- **Path Validation**: Non-existent paths are skipped with a warning

### Error Handling

- Invalid paths are logged and skipped
- Non-existent directories are warned but don't block startup
- URL-like paths (s3://, nfs://, etc.) are rejected with a warning
- If no valid directories are found, falls back to defaults

### Default Behavior

If `XAGENT_SKILLS_LIBRARY_DIRS` is not set or contains no valid paths, the following default directories are used:
1. Built-in skills: `src/xagent/skills/builtin/`
2. User skills: `.xagent/skills/`

### Examples

#### Development Environment
```bash
# Use local skills directory for development
export XAGENT_SKILLS_LIBRARY_DIRS="~/dev/skills,./team_skills"
```

#### Production Environment
```bash
# Use shared skills directory
export XAGENT_SKILLS_LIBRARY_DIRS="/opt/xagent/skills,/shared/company_skills"
```

#### Team Collaboration
```bash
# Combine personal and team skills
export XAGENT_SKILLS_LIBRARY_DIRS="~/my_skills,~/team_skills"
```
