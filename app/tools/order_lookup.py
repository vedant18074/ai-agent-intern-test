import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class OrderLookupResult:
    found: bool
    order_id: Optional[str] = None
    status: Optional[str] = None
    status_updated_at: Optional[str] = None
    carrier: Optional[str] = None
    tracking_number: Optional[str] = None
    estimated_delivery: Optional[str] = None
    customer_safe_message: Optional[str] = None
    message: Optional[str] = None


class OrderLookupService:
    """
    Safe order-status lookup service.

    The service loads orders.json internally and exposes
    only customer-safe order information.

    Internal fields such as:
    - email
    - shipping address
    - risk score
    - warehouse notes
    - support tags

    are never returned.
    """

    ORDER_ID_PATTERN = re.compile(
        r"^ORD-\d+$",
        re.IGNORECASE,
    )

    def __init__(
        self,
        orders_path: str = "data/orders.json",
    ):
        self.orders_path = Path(orders_path)
        self.orders = self._load_orders()

    def _load_orders(self):
        """
        Load the order dataset into memory.

        The complete orders file stays inside the
        application and is never sent to the LLM.
        """

        if not self.orders_path.exists():
            raise FileNotFoundError(
                f"Orders file not found: {self.orders_path}"
            )

        with self.orders_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        orders = data.get("orders")

        if not isinstance(orders, list):
            raise ValueError(
                "Invalid orders.json: 'orders' must be a list."
            )

        return orders

    @classmethod
    def normalize_order_id(
        cls,
        order_id: str,
    ) -> Optional[str]:
        """
        Normalize harmless input differences.

        Examples:

            'ord-1007'  -> 'ORD-1007'
            ' ORD-1007 ' -> 'ORD-1007'
        """

        if not isinstance(order_id, str):
            return None

        normalized = order_id.strip().upper()

        if not cls.ORDER_ID_PATTERN.fullmatch(
            normalized
        ):
            return None

        return normalized

    def lookup(
        self,
        order_id: str,
    ) -> OrderLookupResult:
        """
        Look up an order safely.

        Unknown or malformed IDs never cause an
        invented response.
        """

        normalized_id = self.normalize_order_id(
            order_id
        )

        if normalized_id is None:
            return OrderLookupResult(
                found=False,
                message=(
                    "The order ID format is invalid. "
                    "Please provide an order ID such as "
                    "ORD-1007."
                ),
            )

        for order in self.orders:

            stored_id = str(
                order.get("order_id", "")
            ).strip().upper()

            if stored_id != normalized_id:
                continue

            status = (
                str(order.get("status", ""))
                .strip()
                .lower()
            )

            # --------------------------------------------------
            # Customer-safe fields only
            # --------------------------------------------------

            carrier = order.get("carrier")
            tracking_number = order.get(
                "tracking_number"
            )

            estimated_delivery = order.get(
                "estimated_delivery"
            )

            customer_safe_message = order.get(
                "customer_safe_message"
            )

            # --------------------------------------------------
            # Do NOT expose stale delivery information
            # for cancelled or returned orders.
            # --------------------------------------------------

            if status in {
                "cancelled",
                "canceled",
                "returned",
            }:
                carrier = None
                tracking_number = None
                estimated_delivery = None

            return OrderLookupResult(
                found=True,
                order_id=normalized_id,
                status=status,
                status_updated_at=order.get(
                    "status_updated_at"
                ),
                carrier=carrier,
                tracking_number=tracking_number,
                estimated_delivery=estimated_delivery,
                customer_safe_message=customer_safe_message,
            )

        # --------------------------------------------------
        # Order ID was valid but does not exist.
        # --------------------------------------------------

        return OrderLookupResult(
            found=False,
            order_id=normalized_id,
            message=(
                f"No order was found for {normalized_id}."
            ),
        )