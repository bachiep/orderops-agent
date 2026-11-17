"""Idempotent SQLite demo dataset seeder for OrderOps Agent."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta

from orderops.config import get_settings
from orderops.db import DatabaseManager


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).replace(microsecond=0).isoformat()


def seed_demo_database(
    db: DatabaseManager,
    *,
    now: datetime | None = None,
    reset: bool = False,
) -> dict[str, int]:
    """Initialize SQLite schema and idempotently seed customers, policies, orders, and refunds."""
    db.init_schema()
    ref_now = (now or datetime.now(UTC)).astimezone(UTC).replace(microsecond=0)

    customers = [
        (
            "CUST-001",
            "Nguyen Minh Anh",
            "minh.anh@example.com",
            "0901234567",
            "VIP",
            _iso(ref_now - timedelta(days=120)),
        ),
        (
            "CUST-002",
            "Tran Quoc Bao",
            "quoc.bao@example.com",
            "0912345678",
            "STANDARD",
            _iso(ref_now - timedelta(days=90)),
        ),
        (
            "CUST-003",
            "Le Thu Ha",
            "thu.ha@example.com",
            "0987654321",
            "STANDARD",
            _iso(ref_now - timedelta(days=60)),
        ),
        (
            "CUST-004",
            "Pham Gia Huy",
            "gia.huy@example.com",
            "0938112233",
            "VIP",
            _iso(ref_now - timedelta(days=45)),
        ),
        (
            "CUST-005",
            "Vo Thanh Tam",
            "thanh.tam@example.com",
            "0977445566",
            "STANDARD",
            _iso(ref_now - timedelta(days=30)),
        ),
    ]

    policies = [
        (
            "POL-ELEC-DEF",
            "electronics",
            "defective",
            1,
            14,
            0,
            1,
            "Defective electronics qualify for 100% refund within 14 days of delivery.",
        ),
        (
            "POL-ELEC-WRONG",
            "electronics",
            "wrong_item",
            1,
            14,
            0,
            1,
            "Wrong electronics item shipped qualifies for full refund within 14 days.",
        ),
        (
            "POL-ELEC-DMG",
            "electronics",
            "damaged_in_transit",
            1,
            14,
            0,
            1,
            "Transit-damaged electronics qualify for full refund within 14 days.",
        ),
        (
            "POL-ELEC-MIND",
            "electronics",
            "changed_mind",
            1,
            14,
            10,
            0,
            "Unopened change-of-mind electronics returns incur a 10% restocking fee (<= 14 days).",
        ),
        (
            "POL-APP-ANY",
            "apparel",
            "any",
            1,
            14,
            0,
            0,
            "Apparel items in unworn condition qualify for full refund within 14 days.",
        ),
        (
            "POL-HOME-ANY",
            "home_appliances",
            "any",
            1,
            14,
            0,
            1,
            "Home appliances qualify for return/refund within 14 days of delivery.",
        ),
        (
            "POL-FOOD-ANY",
            "perishable_food",
            "any",
            0,
            0,
            0,
            0,
            "Perishable food and beverage items are strictly non-returnable after delivery.",
        ),
        (
            "POL-VOUCH-ANY",
            "digital_voucher",
            "any",
            0,
            0,
            0,
            0,
            "Digital vouchers and gift codes are non-returnable once issued.",
        ),
        (
            "POL-COSM-DMG",
            "cosmetics",
            "damaged_in_transit",
            1,
            14,
            0,
            1,
            "Cosmetics damaged in transit qualify for refund within 14 days with photo evidence.",
        ),
        (
            "POL-COSM-DEF",
            "cosmetics",
            "defective",
            1,
            14,
            0,
            1,
            "Defective cosmetics qualify for refund within 14 days.",
        ),
        (
            "POL-COSM-MIND",
            "cosmetics",
            "changed_mind",
            0,
            0,
            0,
            0,
            "Change-of-mind cosmetics returns are not accepted for hygiene reasons.",
        ),
    ]

    orders = [
        (
            "ORD-1001",
            "CUST-001",
            "USB-C Mechanical Keycap Set",
            "electronics",
            1,
            320_000,
            "DELIVERED",
            _iso(ref_now - timedelta(days=5)),
            _iso(ref_now - timedelta(days=3)),
            "VNPOST-881001",
        ),
        (
            "ORD-1002",
            "CUST-001",
            "Wireless Noise-Canceling Headphones",
            "electronics",
            1,
            1_850_000,
            "DELIVERED",
            _iso(ref_now - timedelta(days=8)),
            _iso(ref_now - timedelta(days=5)),
            "GHN-881002",
        ),
        (
            "ORD-1003",
            "CUST-002",
            "Ergonomic Desk Lamp",
            "home_appliances",
            1,
            450_000,
            "DELIVERED",
            _iso(ref_now - timedelta(days=25)),
            _iso(ref_now - timedelta(days=22)),
            "GHTK-881003",
        ),
        (
            "ORD-1004",
            "CUST-002",
            "Smart Air Purifier",
            "home_appliances",
            1,
            2_400_000,
            "IN_TRANSIT",
            _iso(ref_now - timedelta(days=2)),
            None,
            "GHN-881004",
        ),
        (
            "ORD-1005",
            "CUST-003",
            "Artisan Drip Coffee Beans 1kg",
            "perishable_food",
            2,
            280_000,
            "DELIVERED",
            _iso(ref_now - timedelta(days=4)),
            _iso(ref_now - timedelta(days=2)),
            "VNPOST-881005",
        ),
        (
            "ORD-1006",
            "CUST-003",
            "E-Gift Voucher 500K",
            "digital_voucher",
            1,
            500_000,
            "DELIVERED",
            _iso(ref_now - timedelta(days=1)),
            _iso(ref_now - timedelta(days=1)),
            "DIGITAL-881006",
        ),
        (
            "ORD-1007",
            "CUST-004",
            "Linen Overshirt",
            "apparel",
            1,
            480_000,
            "REFUNDED",
            _iso(ref_now - timedelta(days=9)),
            _iso(ref_now - timedelta(days=6)),
            "GHTK-881007",
        ),
        (
            "ORD-1008",
            "CUST-004",
            "Mechanical Keyboard Pro",
            "electronics",
            1,
            1_290_000,
            "DELIVERED",
            _iso(ref_now - timedelta(days=13)),
            _iso(ref_now - timedelta(days=10)),
            "GHN-881008",
        ),
        (
            "ORD-1009",
            "CUST-005",
            "Heavyweight Cotton Hoodie",
            "apparel",
            1,
            390_000,
            "DELIVERED",
            _iso(ref_now - timedelta(days=17)),
            _iso(ref_now - timedelta(days=14)),
            "VNPOST-881009",
        ),
        (
            "ORD-1010",
            "CUST-005",
            "Hydrating Facial Serum 50ml",
            "cosmetics",
            1,
            650_000,
            "DELIVERED",
            _iso(ref_now - timedelta(days=6)),
            _iso(ref_now - timedelta(days=4)),
            "GHN-881010",
        ),
        (
            "ORD-1011",
            "CUST-002",
            "Insulated Ceramic Mug Set",
            "home_appliances",
            2,
            500_000,
            "DELIVERED",
            _iso(ref_now - timedelta(days=10)),
            _iso(ref_now - timedelta(days=7)),
            "GHTK-881011",
        ),
        (
            "ORD-1012",
            "CUST-003",
            "Silent Bluetooth Mouse",
            "electronics",
            1,
            250_000,
            "DELIVERED",
            _iso(ref_now - timedelta(days=18)),
            _iso(ref_now - timedelta(days=15)),
            "VNPOST-881012",
        ),
    ]

    seeded_refunds = [
        (
            "RFD-SEED-001",
            "ORD-1007",
            "CUST-004",
            480_000,
            "wrong_item",
            "Received wrong size M instead of L",
            "AUTO_APPROVED",
            0,
            _iso(ref_now - timedelta(days=5)),
        ),
    ]

    with db.transaction() as conn:
        if reset:
            conn.execute("DELETE FROM agent_runs;")
            conn.execute("DELETE FROM refund_requests;")
            conn.execute("DELETE FROM orders;")
            conn.execute("DELETE FROM return_policies;")
            conn.execute("DELETE FROM customers;")

        conn.executemany(
            """
            INSERT OR IGNORE INTO customers (
                customer_id, full_name, email, phone, tier, created_at
            ) VALUES (?, ?, ?, ?, ?, ?);
            """,
            customers,
        )
        conn.executemany(
            """
            INSERT OR IGNORE INTO return_policies (
                policy_id, category, issue_type, is_returnable,
                return_window_days, restocking_fee_pct, requires_photo_evidence, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            policies,
        )
        conn.executemany(
            """
            INSERT OR IGNORE INTO orders (
                order_id, customer_id, item_name, category, quantity,
                total_vnd, status, ordered_at, delivered_at, tracking_code
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            orders,
        )
        conn.executemany(
            """
            INSERT OR IGNORE INTO refund_requests (
                refund_id, order_id, customer_id, amount_vnd,
                reason_code, reason_detail, status, requires_human_review, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            seeded_refunds,
        )

    return db.get_table_counts()


def main() -> None:
    """CLI entrypoint for seeding the demo SQLite database."""
    parser = argparse.ArgumentParser(description="Seed OrderOps Agent demo SQLite database.")
    parser.add_argument("--db-path", default=None, help="Override SQLite database path.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear existing rows before seeding demo data.",
    )
    args = parser.parse_args()

    settings = get_settings()
    db_path = args.db_path or settings.database_path
    with DatabaseManager(db_path) as db:
        counts = seed_demo_database(db, reset=args.reset)
    print(f"Seeded SQLite database at {db_path}: {counts}")


if __name__ == "__main__":
    main()
