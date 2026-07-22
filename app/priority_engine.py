from app.schemas import PriorityEnum
from app.label_maps import normalize_category_label, normalize_sentiment_label

# -----------------------------------------------------------------------------
# Business Rules Configuration (Centralized Sets & Mappings)
# -----------------------------------------------------------------------------

# Critical categories that always bypass sentiment analysis
CRITICAL_CATEGORIES = {
    "System Outage",
    "Security Incident",
}

# High-risk categories: Escalates to High if sentiment is escalated (Negative/Angry)
HIGH_RISK_CATEGORIES = {
    "Hardware Issue",
    "Network & Connectivity",
}

# Medium-risk categories: Escalates to Medium if sentiment is escalated (Negative/Angry)
MEDIUM_RISK_CATEGORIES = {
    "Software Bug",
    "Login & Account Access",
    "Billing & Payments",
    "Orders & Subscription",
}

# Sentiments that trigger priority escalation
ESCALATED_SENTIMENTS = {
    "negative",
    "angry",
}

# Sentiments that trigger priority de-escalation for high-risk categories
RELAXED_SENTIMENTS = {
    "neutral",
    "positive",
}

# Default fallback priorities for every category
DEFAULT_CATEGORY_PRIORITIES: dict[str, PriorityEnum] = {
    "System Outage": PriorityEnum.CRITICAL,
    "Security Incident": PriorityEnum.CRITICAL,
    "Hardware Issue": PriorityEnum.HIGH,
    "Network & Connectivity": PriorityEnum.HIGH,
    "Login & Account Access": PriorityEnum.MEDIUM,
    "Orders & Subscription": PriorityEnum.MEDIUM,
    "Software Bug": PriorityEnum.MEDIUM,
    "Password Reset": PriorityEnum.LOW,
    "Account Management": PriorityEnum.LOW,
    "Billing & Payments": PriorityEnum.LOW,
    "Feature Request": PriorityEnum.LOW,
    "General Inquiry": PriorityEnum.LOW,
}


def calculate_priority(category: str, sentiment: str) -> PriorityEnum:
    """Calculates the deterministic business priority level of a ticket.

    Combines the predicted category and sentiment according to strict business rules.

    Args:
        category (str): The predicted ticket category.
        sentiment (str): The predicted ticket sentiment (e.g., 'Negative',
          'Angry').

    Returns:
        PriorityEnum: The resulting priority level (Critical, High, Medium,
        Low).
    """
    category_clean = normalize_category_label(category)
    sentiment_clean = normalize_sentiment_label(sentiment)

    # Rule 1: Always Critical for severe system conditions
    if category_clean in CRITICAL_CATEGORIES:
        return PriorityEnum.CRITICAL

    # Rule 2: Escalation for High-Risk categories on negative/angry sentiment
    if category_clean in HIGH_RISK_CATEGORIES:
        if sentiment_clean in ESCALATED_SENTIMENTS:
            return PriorityEnum.HIGH

    # Rule 3: Escalation for Medium-Risk categories on negative/angry sentiment
    if category_clean in MEDIUM_RISK_CATEGORIES:
        if sentiment_clean in ESCALATED_SENTIMENTS:
            return PriorityEnum.MEDIUM

    # Rule 4: De-escalation for High-Risk categories on relaxed sentiment
    if sentiment_clean in RELAXED_SENTIMENTS:
        if category_clean in HIGH_RISK_CATEGORIES:
            return PriorityEnum.MEDIUM
        return PriorityEnum.LOW

    # Fallback: Use default priority mapping for category
    return DEFAULT_CATEGORY_PRIORITIES.get(category_clean, PriorityEnum.LOW)