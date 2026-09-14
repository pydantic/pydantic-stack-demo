"""Evaluation dataset and evaluator for contact extraction.

`FieldAccuracyEvaluator` returns a dict: the float `accuracy` becomes a score the optimizer
maximises, and the per-field booleans become assertions you can read in the report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from prompt_optimization.task import ContactInfo, TaskInput

FIELDS = ('name', 'email', 'phone', 'company', 'title')


def _normalise(value: object) -> str:
    return ' '.join(str(value or '').lower().split())


@dataclass
class ContactCaseMetadata:
    difficulty: str  # 'easy', 'medium', 'hard'
    has_noise: bool  # whether the text contains irrelevant information
    description: str


@dataclass
class FieldAccuracyEvaluator(Evaluator[TaskInput, ContactInfo, ContactCaseMetadata]):
    """Fraction of expected fields extracted correctly, plus one assertion per field."""

    def evaluate(self, ctx: EvaluatorContext[TaskInput, ContactInfo, ContactCaseMetadata]) -> dict[str, Any]:
        if ctx.expected_output is None:
            return {'accuracy': 1.0}

        results: dict[str, Any] = {}
        correct = total = 0
        for field in FIELDS:
            expected = getattr(ctx.expected_output, field)
            if expected is None:
                continue
            total += 1
            actual = getattr(ctx.output, field)
            # Exact match after normalising case and whitespace: "Dr. Maria Garcia" is *not* "Maria Garcia",
            # which is precisely the kind of rule a better prompt has to spell out.
            is_correct = _normalise(expected) == _normalise(actual)
            correct += is_correct
            results[f'{field}_correct'] = is_correct

        results['accuracy'] = correct / total if total else 1.0
        results['fields_correct'] = correct
        results['fields_total'] = total
        return results


contact_cases: list[Case[TaskInput, ContactInfo, ContactCaseMetadata]] = [
    Case(
        name='simple_email_signature',
        inputs=TaskInput(text='Best regards,\nJohn Smith\njohn.smith@example.com\n555-123-4567'),
        expected_output=ContactInfo(name='John Smith', email='john.smith@example.com', phone='555-123-4567'),
        metadata=ContactCaseMetadata(difficulty='easy', has_noise=False, description='Simple email signature'),
    ),
    Case(
        name='business_card_format',
        inputs=TaskInput(
            text='Jane Doe\nSenior Software Engineer\nAcme Corporation\njane.doe@acme.com\n(415) 555-0100'
        ),
        expected_output=ContactInfo(
            name='Jane Doe',
            email='jane.doe@acme.com',
            phone='(415) 555-0100',
            company='Acme Corporation',
            title='Senior Software Engineer',
        ),
        metadata=ContactCaseMetadata(difficulty='easy', has_noise=False, description='Standard business card'),
    ),
    Case(
        name='inline_contact',
        inputs=TaskInput(
            text='For more information, contact Michael Johnson at michael.j@techcorp.io or call him at 1-800-TECH-123.'
        ),
        expected_output=ContactInfo(name='Michael Johnson', email='michael.j@techcorp.io', phone='1-800-TECH-123'),
        metadata=ContactCaseMetadata(difficulty='medium', has_noise=True, description='Contact embedded in a sentence'),
    ),
    Case(
        name='international_format',
        inputs=TaskInput(
            text='Dr. Maria Garcia\nHead of Research\nGlobal Health Institute\nmgarcia@ghi.org\n+44 20 7946 0958'
        ),
        expected_output=ContactInfo(
            name='Maria Garcia',
            email='mgarcia@ghi.org',
            phone='+44 20 7946 0958',
            company='Global Health Institute',
            title='Head of Research',
        ),
        metadata=ContactCaseMetadata(
            difficulty='medium', has_noise=False, description='International phone, title prefix'
        ),
    ),
    Case(
        name='noisy_email_thread',
        inputs=TaskInput(
            text="""Re: Meeting Follow-up

Hi team,

Thanks for joining today's call. As discussed, please reach out to our new vendor contact:

Robert Chen
VP of Sales, CloudTech Solutions
r.chen@cloudtech.solutions
Mobile: +1 (650) 555-8900

He'll be handling our account going forward. The previous contact (sarah@oldvendor.com) is no longer with the company.

Best,
Alex"""
        ),
        expected_output=ContactInfo(
            name='Robert Chen',
            email='r.chen@cloudtech.solutions',
            phone='+1 (650) 555-8900',
            company='CloudTech Solutions',
            title='VP of Sales',
        ),
        metadata=ContactCaseMetadata(difficulty='hard', has_noise=True, description='Two contacts; pick the main one'),
    ),
    Case(
        name='partial_info',
        inputs=TaskInput(text='Please contact support at help@service.io for assistance.'),
        expected_output=ContactInfo(email='help@service.io'),
        metadata=ContactCaseMetadata(difficulty='medium', has_noise=False, description='Only an email address'),
    ),
    Case(
        name='informal_intro',
        inputs=TaskInput(text="Hey! I'm Sam from StartupXYZ. Hit me up at sam@startupxyz.com or text 555-STARTUP"),
        expected_output=ContactInfo(name='Sam', email='sam@startupxyz.com', phone='555-STARTUP', company='StartupXYZ'),
        metadata=ContactCaseMetadata(difficulty='medium', has_noise=False, description='Informal, vanity phone number'),
    ),
    Case(
        name='complex_signature',
        inputs=TaskInput(
            text="""--
Emily Watson, Ph.D.
Director of Engineering | AI Division
TechGiant Inc. (NASDAQ: TGNT)

Email: e.watson@techgiant.com
Office: (555) 100-2000 ext. 4501
LinkedIn: linkedin.com/in/emilywatson

"Innovation distinguishes between a leader and a follower."
This email may contain confidential information..."""
        ),
        expected_output=ContactInfo(
            name='Emily Watson',
            email='e.watson@techgiant.com',
            phone='(555) 100-2000 ext. 4501',
            company='TechGiant Inc.',
            title='Director of Engineering',
        ),
        metadata=ContactCaseMetadata(difficulty='hard', has_noise=True, description='Credentials, ticker and noise'),
    ),
]

contact_dataset: Dataset[TaskInput, ContactInfo, ContactCaseMetadata] = Dataset(
    name='contact_extraction',
    cases=contact_cases,
    evaluators=[FieldAccuracyEvaluator()],
)
