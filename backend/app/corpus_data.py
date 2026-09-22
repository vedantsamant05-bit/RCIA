"""
corpus_data.py
--------------
Synthetic-but-realistic internal policy corpus (Section 4.1 of the project
doc). Real internal bank/fintech policies aren't accessible, so this is a
hand-built corpus mimicking real SOP/contract structure, deliberately
seeded with clauses that overlap (fully, partially, or not at all) with
the sample regulations below -- so retrieval and impact-classification
can actually be tested against known answers.

Also includes a small labeled eval set (Section 5) so the /api/eval
endpoint has something real to score against.
"""

# Each internal "document" is a list of (section, clause_text) tuples.
INTERNAL_POLICIES = {
    "Customer Data Retention Policy v3.2": [
        ("2.1", "Customer personally identifiable information (PII) collected during onboarding shall be retained for a period of ten (10) years from the date of account closure."),
        ("2.2", "Transaction records shall be retained for a minimum of five (5) years and may be purged thereafter at the discretion of the Data Governance team."),
        ("3.1", "Customers may request a copy of their personal data by submitting a written request to the branch manager, to be fulfilled within thirty (30) business days."),
        ("3.2", "The Bank does not currently provide a mechanism for customers to request deletion of their personal data prior to the standard retention period."),
        ("4.1", "Data breach incidents affecting customer PII shall be reported to the Chief Information Security Officer within seventy-two (72) hours of detection."),
    ],
    "KYC and Customer Onboarding SOP v5.0": [
        ("1.1", "All new individual customers must provide a valid government-issued photo identification and proof of address prior to account opening."),
        ("1.2", "Enhanced due diligence shall be performed for customers classified as Politically Exposed Persons (PEPs), including verification of source of funds."),
        ("1.3", "Video-based KYC (V-CIP) may be used as an alternative to in-branch verification for customers opening savings accounts, subject to a maximum daily transaction limit as defined in Annexure A."),
        ("2.1", "Periodic KYC updation shall be conducted every ten (10) years for low-risk customers and every two (2) years for high-risk customers."),
        ("2.2", "Re-KYC may be waived for customers who have had no change in registered address or contact details since the last verification cycle."),
    ],
    "Third-Party Vendor Data Sharing Agreement Template": [
        ("5.1", "Vendor may process customer data solely for the purposes explicitly enumerated in Schedule B of this Agreement."),
        ("5.2", "Vendor shall not transfer customer data outside the jurisdiction in which it was originally collected without prior written consent from the Bank."),
        ("6.1", "Vendor shall implement encryption at rest and in transit for all customer data processed under this Agreement, using industry-standard algorithms."),
        ("7.1", "In the event of a sub-processing arrangement, Vendor shall notify the Bank at least fifteen (15) business days prior to onboarding a new sub-processor."),
    ],
    "Consumer Loan Agreement Standard Terms": [
        ("3.1", "The Borrower shall be provided with a Key Facts Statement summarizing the annualized percentage rate, all applicable fees, and the total cost of credit prior to loan disbursal."),
        ("3.2", "Prepayment of the loan, in part or in full, shall attract a foreclosure charge of up to four percent (4%) of the outstanding principal, except where the loan carries a floating rate of interest."),
        ("4.1", "Any change to the interest rate applicable to a floating-rate loan shall be communicated to the Borrower at least thirty (30) days prior to the change taking effect."),
        ("5.1", "In the event of default, the Lender may report the Borrower's account status to credit information companies without further notice to the Borrower."),
    ],
    "Digital Lending Grievance Redressal Policy": [
        ("1.1", "Grievances relating to digital lending products shall be resolved within thirty (30) days of receipt."),
        ("1.2", "The name and contact details of the Grievance Redressal Officer shall be prominently displayed on the lending application's home screen and in the loan agreement."),
        ("2.1", "Where a grievance is not resolved within the stipulated timeline, the customer may escalate the matter to the Banking Ombudsman."),
    ],
    "Employee Data Access and Least-Privilege Policy": [
        ("1.1", "Access to customer PII by internal staff shall be granted strictly on a need-to-know basis, tied to the employee's active role and function."),
        ("1.2", "Access logs for systems containing customer PII shall be retained for a minimum of one (1) year and reviewed quarterly by Internal Audit."),
        ("2.1", "Employees separating from the organization shall have all system access revoked within twenty-four (24) hours of their last working day."),
    ],
    "Marketing Consent and Communication Policy": [
        ("1.1", "Customers shall be presumed to have consented to marketing communications unless they have explicitly opted out via the registered channel."),
        ("1.2", "Marketing communications shall not be sent to customers who have registered on the National Do Not Call (NDNC) registry, except where explicit consent has been separately obtained."),
        ("2.1", "Consent records for marketing communications shall be maintained for the duration of the customer relationship plus one (1) year thereafter."),
    ],
    "Cross-Border Data Transfer Standard": [
        ("1.1", "Customer data may be transferred to a group entity located outside the country of collection only where that entity is bound by a Bank-approved data processing addendum."),
        ("1.2", "Critical personal data, as classified under Annexure C, shall be processed and stored exclusively within the country of collection."),
        ("2.1", "A record of all cross-border data transfers shall be maintained and made available to the regulator upon request within seven (7) business days."),
    ],
}

# Sample public-style regulatory texts (illustrative paraphrases of the
# kind of obligations found in real circulars -- written for this demo,
# not verbatim reproductions of any single regulator's text).
SAMPLE_REGULATIONS = [
    {
        "title": "Data Protection Amendment - Right to Erasure",
        "source": "DPDP Act - Illustrative Amendment Notice",
        "text": (
            "Clause 4(a): Every data fiduciary shall, upon receiving a verified request from a data principal, "
            "erase the personal data of that data principal within thirty (30) days, unless retention is "
            "required under any other law for the time being in force.\n\n"
            "Clause 4(b): Data fiduciaries shall implement a mechanism through which a data principal may submit "
            "a request for erasure of personal data through the same channel used for account services, "
            "without requiring an in-person visit.\n\n"
            "Clause 5(a): Where personal data is transferred to a processor located outside the territory, the "
            "data fiduciary shall ensure that such transfer is accompanied by contractual safeguards no weaker "
            "than those required for domestic processing, and shall maintain a transfer log accessible to the "
            "Board within five (5) business days of a request."
        ),
    },
    {
        "title": "Digital Lending Directions - Amendment on Rate Reset Notice",
        "source": "RBI - Illustrative Circular",
        "text": (
            "Para 3.1: Regulated Entities offering floating-rate retail loans shall notify borrowers of any "
            "change in the applicable interest rate at least fifteen (15) days in advance of the change taking "
            "effect, through the borrower's registered communication channel.\n\n"
            "Para 3.2: Regulated Entities shall not levy a foreclosure or prepayment charge on any floating-rate "
            "retail loan availed by an individual borrower, regardless of the source of funds used for "
            "prepayment.\n\n"
            "Para 4.1: The name, designation, and direct contact details of the Grievance Redressal Officer "
            "shall be displayed on every page of the digital lending application, not solely the home screen."
        ),
    },
    {
        "title": "Master Direction on Periodic KYC Updation",
        "source": "RBI - Illustrative Master Direction Update",
        "text": (
            "Para 2.1: Re-KYC for low-risk individual customers shall be carried out at intervals not exceeding "
            "eight (8) years from the date of the last KYC verification.\n\n"
            "Para 2.2: Regulated Entities shall not waive periodic re-KYC solely on the basis of no change in "
            "address; a positive confirmation of customer details shall be independently obtained at each "
            "re-KYC cycle regardless of address change status."
        ),
    },
]

# Section 5: hand-labeled (regulation clause, internal clause) pairs used
# by the /api/eval endpoint to compute retrieval/classification metrics.
# is_impacted: 1 if the regulation clause should flag the internal clause.
EVAL_LABELS = [
    {
        "regulation_clause_text": "Every data fiduciary shall, upon receiving a verified request from a data principal, erase the personal data of that data principal within thirty (30) days",
        "internal_doc": "Customer Data Retention Policy v3.2",
        "internal_section": "3.2",
        "is_impacted": 1,
        "true_impact_type": "contradicts",
    },
    {
        "regulation_clause_text": "Regulated Entities shall not levy a foreclosure or prepayment charge on any floating-rate retail loan availed by an individual borrower",
        "internal_doc": "Consumer Loan Agreement Standard Terms",
        "internal_section": "3.2",
        "is_impacted": 1,
        "true_impact_type": "contradicts",
    },
    {
        "regulation_clause_text": "Regulated Entities offering floating-rate retail loans shall notify borrowers of any change in the applicable interest rate at least fifteen (15) days in advance",
        "internal_doc": "Consumer Loan Agreement Standard Terms",
        "internal_section": "4.1",
        "is_impacted": 1,
        "true_impact_type": "tightens",
    },
    {
        "regulation_clause_text": "Re-KYC for low-risk individual customers shall be carried out at intervals not exceeding eight (8) years",
        "internal_doc": "KYC and Customer Onboarding SOP v5.0",
        "internal_section": "2.1",
        "is_impacted": 1,
        "true_impact_type": "tightens",
    },
    {
        "regulation_clause_text": "Regulated Entities shall not waive periodic re-KYC solely on the basis of no change in address",
        "internal_doc": "KYC and Customer Onboarding SOP v5.0",
        "internal_section": "2.2",
        "is_impacted": 1,
        "true_impact_type": "contradicts",
    },
    {
        "regulation_clause_text": "The name, designation, and direct contact details of the Grievance Redressal Officer shall be displayed on every page of the digital lending application",
        "internal_doc": "Digital Lending Grievance Redressal Policy",
        "internal_section": "1.2",
        "is_impacted": 1,
        "true_impact_type": "tightens",
    },
    {
        "regulation_clause_text": "Every data fiduciary shall, upon receiving a verified request from a data principal, erase the personal data of that data principal within thirty (30) days",
        "internal_doc": "Marketing Consent and Communication Policy",
        "internal_section": "2.1",
        "is_impacted": 0,
        "true_impact_type": "no-op",
    },
    {
        "regulation_clause_text": "Where personal data is transferred to a processor located outside the territory, the data fiduciary shall ensure that such transfer is accompanied by contractual safeguards",
        "internal_doc": "Cross-Border Data Transfer Standard",
        "internal_section": "1.1",
        "is_impacted": 1,
        "true_impact_type": "no-op",
    },
    {
        "regulation_clause_text": "Regulated Entities shall not levy a foreclosure or prepayment charge on any floating-rate retail loan",
        "internal_doc": "Customer Data Retention Policy v3.2",
        "internal_section": "2.1",
        "is_impacted": 0,
        "true_impact_type": "no-op",
    },
    {
        "regulation_clause_text": "data fiduciaries shall implement a mechanism through which a data principal may submit a request for erasure of personal data through the same channel used for account services",
        "internal_doc": "Customer Data Retention Policy v3.2",
        "internal_section": "3.1",
        "is_impacted": 1,
        "true_impact_type": "tightens",
    },
]
