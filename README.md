# DREU Research Log

**Student:** Vilohith Gokarakonda
**Mentor:** Dr.Ming C Lin

Hi, I am not sure if you will look at this message, but I started filling out the logs during the weekend and today, since I got the email about them from
jforbes@cra.org on July 3rd. So I filled out the weeks and progression to best of ability from recolection. The logs have a link to where you can see the latex document and I will update my Ipad notes by uploading them to the repo manually as PDFs. I sent a Week 2 Progress Report on 6/22 to dreu_staff@cra.org and Ashley with my notes since I was unaware about the progress report submissions and weekly documentation until 7/3/26. If you have any questions, please let me know at vilohithg@gmail.com.

This repository contains your weekly research log for the Distributed Research Experiences for Undergraduates (DREU) program. It covers 10 weeks. All 10 log files are pre-created — you fill them in week by week.

_Update the Student and Mentor fields above after forking._

## Repository Structure

```
your-repo/
├── README.md          ← this file
├── TEMPLATE.md        ← reference template (do not edit)
├── EXAMPLE.md         ← filled-out example showing expected quality
└── logs/
    ├── week-01.md
    ├── week-02.md
    ├── ...
    └── week-10.md
```

## How to Submit Your Weekly Log

1. Open the current week's file in `logs/` (e.g., `logs/week-03.md`).
2. Fill in the **Dates** field and all three sections.
3. Commit and push your changes before the weekly deadline (Sunday 11:59 PM).

You do not need to create new files — all 10 are already in the repo waiting for you.

## Log Format Requirements

Each weekly log **must** follow this exact structure for automated validation to pass:

```markdown
# Week N

**Dates:** MM-DD to MM-DD

## Goals

...

## Approach and Implementation

...

## Results

...
```

The following fields are required and must appear exactly as written:

| Field      | Example                          |
| ---------- | -------------------------------- |
| H1 title   | `# Week 3`                       |
| Dates line | `**Dates:** 06-16 to 06-22`      |
| Section 1  | `## Goals`                       |
| Section 2  | `## Approach and Implementation` |
| Section 3  | `## Results`                     |

The first three sections are required and must each contain at least one non-empty line of content. A fourth section, `## Notes`, is optional — include it for questions, follow-up items, or anything else that doesn't fit the main sections.

See **EXAMPLE.md** for a complete filled-out sample.

## Checking Your Log

You can validate your own log at any time by running the check script locally:

```bash
python check_log.py logs/week-NN.md
```

Or push your changes — if a GitHub Action is configured in this repo, it will automatically validate your latest log and show a pass/fail status on your commit.

## Questions?

Contact the DREU program staff <dreu_staff@cra.org> if you have questions about the format or submission process.
