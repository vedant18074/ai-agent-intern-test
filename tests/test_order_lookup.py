from app.tools.order_lookup import OrderLookupService


def test_valid_order_lookup():
    service = OrderLookupService()

    result = service.lookup("ORD-1003")

    assert result.found is True
    assert result.order_id == "ORD-1003"
    assert result.status == "shipped"


def test_lowercase_order_id():
    service = OrderLookupService()

    result = service.lookup("ord-1003")

    assert result.found is True
    assert result.order_id == "ORD-1003"


def test_whitespace_order_id():
    service = OrderLookupService()

    result = service.lookup("  ORD-1003  ")

    assert result.found is True
    assert result.order_id == "ORD-1003"


def test_unknown_order_id():
    service = OrderLookupService()

    result = service.lookup("ORD-9999")

    assert result.found is False
    assert result.order_id == "ORD-9999"


def test_malformed_order_id():
    service = OrderLookupService()

    result = service.lookup("ABC-123")

    assert result.found is False


def test_empty_order_id():
    service = OrderLookupService()

    result = service.lookup("")

    assert result.found is False


def test_internal_fields_are_not_exposed():
    service = OrderLookupService()

    result = service.lookup("ORD-1003")

    assert not hasattr(result, "email")
    assert not hasattr(result, "shipping_address")
    assert not hasattr(result, "risk_score")
    assert not hasattr(result, "warehouse_note")
    assert not hasattr(result, "support_tags")


def test_customer_safe_message_is_returned():
    service = OrderLookupService()

    result = service.lookup("ORD-1003")

    assert result.customer_safe_message is not None


def test_missing_estimated_delivery_is_preserved_as_none():
    service = OrderLookupService()

    result = service.lookup("ORD-1001")

    assert result.found is True
    assert result.status == "pending"
    assert result.estimated_delivery is None