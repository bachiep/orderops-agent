"""SQLite schema DDL definitions for OrderOps Agent."""

SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS customers (
        customer_id TEXT PRIMARY KEY,
        full_name TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE,
        phone TEXT NOT NULL,
        tier TEXT NOT NULL DEFAULT 'STANDARD',
        created_at TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS return_policies (
        policy_id TEXT PRIMARY KEY,
        category TEXT NOT NULL,
        issue_type TEXT NOT NULL,
        is_returnable INTEGER NOT NULL CHECK (is_returnable IN (0, 1)),
        return_window_days INTEGER NOT NULL CHECK (return_window_days >= 0),
        restocking_fee_pct INTEGER NOT NULL DEFAULT 0
            CHECK (restocking_fee_pct >= 0 AND restocking_fee_pct <= 100),
        requires_photo_evidence INTEGER NOT NULL DEFAULT 0
            CHECK (requires_photo_evidence IN (0, 1)),
        notes TEXT NOT NULL,
        UNIQUE (category, issue_type)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS orders (
        order_id TEXT PRIMARY KEY,
        customer_id TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE RESTRICT,
        item_name TEXT NOT NULL,
        category TEXT NOT NULL,
        quantity INTEGER NOT NULL DEFAULT 1 CHECK (quantity > 0),
        total_vnd INTEGER NOT NULL CHECK (total_vnd > 0),
        status TEXT NOT NULL CHECK (
            status IN (
                'PROCESSING',
                'IN_TRANSIT',
                'DELIVERED',
                'REFUND_PENDING',
                'REFUNDED',
                'CANCELLED'
            )
        ),
        ordered_at TEXT NOT NULL,
        delivered_at TEXT,
        tracking_code TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS refund_requests (
        refund_id TEXT PRIMARY KEY,
        order_id TEXT NOT NULL REFERENCES orders(order_id) ON DELETE RESTRICT,
        customer_id TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE RESTRICT,
        amount_vnd INTEGER NOT NULL CHECK (amount_vnd > 0),
        reason_code TEXT NOT NULL,
        reason_detail TEXT NOT NULL,
        status TEXT NOT NULL CHECK (
            status IN ('AUTO_APPROVED', 'PENDING_HUMAN_APPROVAL', 'APPROVED', 'REJECTED')
        ),
        requires_human_review INTEGER NOT NULL CHECK (requires_human_review IN (0, 1)),
        created_at TEXT NOT NULL,
        reviewed_at TEXT,
        reviewer_note TEXT,
        reviewer_id TEXT
    );
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_active_refund_order
    ON refund_requests(order_id)
    WHERE status IN ('AUTO_APPROVED', 'PENDING_HUMAN_APPROVAL', 'APPROVED');
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_orders_customer_id
    ON orders(customer_id, ordered_at DESC);
    """,
    """
    CREATE TABLE IF NOT EXISTS agent_runs (
        run_id TEXT PRIMARY KEY,
        session_id TEXT,
        customer_email_masked TEXT,
        user_message_masked TEXT NOT NULL,
        intent TEXT NOT NULL,
        resolution_status TEXT NOT NULL,
        refund_id TEXT,
        requires_human_review INTEGER NOT NULL CHECK (requires_human_review IN (0, 1)),
        guardrail_flags_json TEXT NOT NULL,
        tool_trace_json TEXT NOT NULL,
        steps_taken INTEGER NOT NULL,
        latency_ms REAL NOT NULL,
        total_tokens INTEGER NOT NULL,
        estimated_cost_usd REAL NOT NULL,
        created_at TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS session_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        run_id TEXT REFERENCES agent_runs(run_id) ON DELETE SET NULL,
        role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
        content_masked TEXT NOT NULL,
        intent TEXT,
        resolution_status TEXT,
        refund_id TEXT,
        created_at TEXT NOT NULL
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_session_messages_session_created
    ON session_messages(session_id, created_at ASC, id ASC);
    """,
    """
    CREATE TABLE IF NOT EXISTS refund_audit_logs (
        audit_id TEXT PRIMARY KEY,
        refund_id TEXT NOT NULL REFERENCES refund_requests(refund_id) ON DELETE RESTRICT,
        order_id TEXT NOT NULL REFERENCES orders(order_id) ON DELETE RESTRICT,
        decision TEXT NOT NULL CHECK (decision IN ('APPROVE', 'REJECT')),
        previous_status TEXT NOT NULL,
        new_status TEXT NOT NULL,
        reviewer_id TEXT NOT NULL,
        reviewer_note_masked TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_refund_audit_logs_refund_id
    ON refund_audit_logs(refund_id, created_at DESC);
    """,
)
