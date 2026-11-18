"""A made-up company handbook (Northwind Labs) rendered to PDF, plus an eval set for it.
Questions mix exact terms (where keyword search shines) with paraphrases (where vectors help),
and include questions the handbook can't answer."""

from __future__ import annotations

import json
from pathlib import Path

SECTIONS = [
    ("1 Welcome to Northwind Labs", [
        "Northwind Labs builds data tools for small hospitals. This handbook explains how we work, what we offer "
        "employees, and who to ask when something is unclear.",
        "The handbook applies to all full-time employees in the United States. Contractors should follow the "
        "terms in their own agreements.",
    ]),
    ("2 Working Hours and Remote Work", [
        "Our core collaboration hours are 10am to 3pm Mountain Time, Monday through Friday. Outside core hours "
        "you may arrange your schedule as your team agrees.",
        "Employees may work remotely up to three days per week. Fully remote arrangements need written approval "
        "from both your manager and the People team.",
        "Remote employees receive a one-time home office stipend of 600 dollars, claimable within the first 90 "
        "days of employment.",
    ]),
    ("3 Paid Time Off", [
        "Full-time employees accrue 20 days of paid time off per year, credited monthly. Up to 5 unused days carry "
        "over into the next calendar year; the rest expire on December 31.",
        "Requests for 3 or more consecutive days should be submitted at least two weeks in advance through the "
        "Timekeeper system.",
        "Northwind observes 11 company holidays, including the day after Thanksgiving. The holiday calendar is "
        "published every November.",
    ]),
    ("4 Parental and Medical Leave", [
        "Birthing parents receive 16 weeks of fully paid parental leave. Non-birthing parents receive 12 weeks "
        "of fully paid leave, to be taken within the first year after the birth or adoption.",
        "Employees may take up to 10 paid sick days per year. Sick days do not carry over and are separate from "
        "paid time off.",
    ]),
    ("5 Expenses and Travel", [
        "Business expenses must be submitted in Ledgerly within 30 days with an itemized receipt. Expenses under "
        "25 dollars do not need a receipt.",
        "Economy class is required for flights under six hours. For longer flights, premium economy may be booked "
        "with director approval.",
        "The daily meal allowance during business travel is 75 dollars. Alcohol is not reimbursable.",
    ]),
    ("6 Equipment and Security", [
        "Every employee receives a company laptop, which must use full-disk encryption and the approved password "
        "manager. Lost or stolen devices must be reported to IT within 24 hours.",
        "Patient data may only be accessed on company devices and never stored in personal cloud accounts. "
        "Security training is mandatory every year, and must be completed within 30 days of starting.",
    ]),
    ("7 Learning and Growth", [
        "Each employee has an annual learning budget of 1,500 dollars for courses, books and conferences. "
        "Unused budget does not roll over.",
        "Performance reviews happen twice a year, in April and October. Promotions are decided in the October cycle.",
    ]),
    ("8 Other Leave", [
        "Employees receive up to 3 days of paid bereavement leave for the loss of an immediate family member, "
        "and 1 day for extended family.",
        "Jury duty is paid in full for up to 10 working days per year. Please share your summons with the People "
        "team as soon as you receive it.",
        "Every employee may take 2 paid volunteer days per year to support a registered charity.",
    ]),
    ("9 On-Call and Overtime", [
        "Engineers on the on-call rotation receive 300 dollars per on-call week, plus one day off after any week "
        "with an overnight incident.",
        "Non-exempt employees are paid 1.5 times their hourly rate for hours worked beyond 40 in a week. Overtime "
        "must be approved in advance by a manager.",
    ]),
    ("10 Referrals and Hiring", [
        "Employees who refer a candidate who is hired receive a referral bonus of 2,000 dollars, paid after the "
        "new hire completes 90 days.",
        "Interviewers must complete interview training before joining a hiring panel, and submit written "
        "feedback within 24 hours of each interview.",
    ]),
    ("11 Equipment Refresh and Travel Insurance", [
        "Laptops are refreshed every three years, or sooner if IT confirms a hardware fault. Monitors and "
        "keyboards can be ordered from the IT catalog at any time.",
        "Business travel is covered by the company travel insurance policy, which includes emergency medical "
        "care abroad. Personal trips added to a business trip are not covered.",
        "Conference travel needs approval from your director at least 30 days before the event.",
    ]),
    ("12 Leaving Northwind", [
        "We ask for two weeks of notice when you resign. Your final paycheck includes any accrued, unused paid "
        "time off.",
        "Company equipment must be returned within 7 days of your last day; IT will send a prepaid shipping label.",
    ]),
]

EVALS = [
    {"q": "How many days of PTO do full-time employees get per year?", "pages": [4], "must": ["20"]},
    {"q": "Can I carry over vacation days I didn't use?", "pages": [4], "must": ["5"]},
    {"q": "How far ahead do I need to request a week off?", "pages": [4], "must": ["two weeks"]},
    {"q": "How long is parental leave for a non-birthing parent?", "pages": [5], "must": ["12 weeks"]},
    {"q": "What are the core hours when everyone should be online?", "pages": [3], "must": ["10am", "3pm"]},
    {"q": "How many days a week can I work from home?", "pages": [3], "must": ["three"]},
    {"q": "Is there money to set up a home office?", "pages": [3], "must": ["600"]},
    {"q": "What is the deadline for submitting expenses in Ledgerly?", "pages": [6], "must": ["30 days"]},
    {"q": "Do I need a receipt for a 15 dollar lunch?", "pages": [6], "must": ["25"]},
    {"q": "How much can I spend on food per day when traveling?", "pages": [6], "must": ["75"]},
    {"q": "What should I do if my laptop gets stolen?", "pages": [7], "must": ["24 hours"]},
    {"q": "How much is the yearly budget for courses and conferences?", "pages": [8], "must": ["1,500"]},
    {"q": "When are promotions decided?", "pages": [8], "must": ["October"]},
    {"q": "How soon must I return my laptop after quitting?", "pages": [13], "must": ["7 days"]},
    {"q": "How many paid days off do I get when a parent dies?", "pages": [9], "must": ["3"]},
    {"q": "What do engineers get paid for an on-call week?", "pages": [10], "must": ["300"]},
    {"q": "When is the referral bonus paid out?", "pages": [11], "must": ["90 days"]},
    {"q": "How often do we get a new laptop?", "pages": [12], "must": ["three years"]},
    {"q": "How early must a conference trip be approved?", "pages": [12], "must": ["30 days"]},
    {"q": "Am I covered if I get sick abroad on a work trip?", "pages": [12], "must": ["emergency medical"]},
    {"q": "What is the company's 401(k) match?", "pages": []},
    {"q": "Does Northwind offer stock options to new hires?", "pages": []},
    {"q": "What is the dress code for the office?", "pages": []},
    {"q": "Who is the CEO of Northwind Labs?", "pages": []},
]


def write_pdf(dest: Path) -> Path:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

    styles = getSampleStyleSheet()
    story = [Paragraph("Northwind Labs Employee Handbook", styles["Title"]),
             Paragraph("Version 2026.2. For internal use.", styles["Normal"]), PageBreak()]
    for title, paras in SECTIONS:
        story.append(Paragraph(title, styles["Heading1"]))
        for p in paras:
            story += [Paragraph(p, styles["BodyText"]), Spacer(1, 8)]
        story.append(PageBreak())
    dest.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(dest), pagesize=letter).build(story)
    return dest


def write_evals(dest: Path) -> Path:
    dest.write_text("\n".join(json.dumps(e) for e in EVALS) + "\n")
    return dest
