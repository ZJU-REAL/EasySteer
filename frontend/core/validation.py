"""Request payload validation helpers shared by the API blueprints."""

from .messages import get_message


def require_fields(data, fields, lang='zh'):
    """Check that each field is present and non-empty in the request payload.

    A field is considered missing when it is absent or its value is None,
    an empty string, or an empty list (numeric 0 is a valid value).

    Returns:
        The localized error message for the first missing field, or None
        when all fields are present.
    """
    for field in fields:
        if data is None or field not in data or data[field] in (None, '', []):
            return get_message('missing_field', lang, field=field)
    return None
