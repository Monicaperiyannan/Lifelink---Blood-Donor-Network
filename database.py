import sqlite3
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash
from config import Config


def get_db_connection():
    conn = sqlite3.connect(Config.DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def create_tables():
    conn = get_db_connection()
    c = conn.cursor()

    # ── Users table ───────────────────────────────────────
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT    NOT NULL UNIQUE,
            email         TEXT    NOT NULL UNIQUE,
            password_hash TEXT    NOT NULL,
            full_name     TEXT,
            phone         TEXT,
            blood_group   TEXT,
            city          TEXT,
            role          TEXT    NOT NULL DEFAULT 'user',
            is_blocked    INTEGER DEFAULT 0,
            profile_complete INTEGER DEFAULT 0,
            created_at    TEXT    DEFAULT (datetime('now'))
        )
    ''')

    # ── Donors table ──────────────────────────────────────
    c.execute('''
        CREATE TABLE IF NOT EXISTS donors (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id       INTEGER,
            full_name     TEXT    NOT NULL,
            blood_group   TEXT    NOT NULL,
            age           INTEGER NOT NULL,
            gender        TEXT    NOT NULL,
            weight        REAL    NOT NULL,
            phone         TEXT    NOT NULL UNIQUE,
            email         TEXT    UNIQUE,
            city          TEXT    NOT NULL,
            state         TEXT,
            is_available  INTEGER DEFAULT 1,
            last_donated  TEXT,
            created_at    TEXT    DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    # ── Emergency Requests table ──────────────────────────
    c.execute('''
        CREATE TABLE IF NOT EXISTS emergency_requests (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id       INTEGER,
            patient_name  TEXT    NOT NULL,
            blood_group   TEXT    NOT NULL,
            hospital      TEXT    NOT NULL,
            district      TEXT    NOT NULL,
            contact_phone TEXT    NOT NULL,
            units_needed  INTEGER DEFAULT 1,
            status        TEXT    DEFAULT 'pending',
            admin_notes   TEXT,
            notes         TEXT,
            created_at    TEXT    DEFAULT (datetime('now')),
            updated_at    TEXT    DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    # ── Announcements table ───────────────────────────────
    c.execute('''
        CREATE TABLE IF NOT EXISTS announcements (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            title      TEXT NOT NULL,
            content    TEXT NOT NULL,
            is_active  INTEGER DEFAULT 1,
            created_at TEXT DEFAULT (datetime('now'))
        )
    ''')

    conn.commit()

    # Create default admin
    c.execute("SELECT COUNT(*) as cnt FROM users WHERE role='admin'")
    if c.fetchone()['cnt'] == 0:
        c.execute('''
            INSERT INTO users (username, email, password_hash, full_name, role)
            VALUES (?, ?, ?, ?, ?)
        ''', ('admin', 'admin@lifelink.health',
              generate_password_hash('admin123'), 'LifeLink Admin', 'admin'))
        conn.commit()
        print("✅ Admin created: username=admin | password=admin123")

    conn.close()
    print("✅ All tables ready.")


# ══════════════════════════════════════════════════════════
# USER FUNCTIONS
# ══════════════════════════════════════════════════════════

def create_user(username, email, password, full_name='', role='user'):
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute('''
            INSERT INTO users (username, email, password_hash, full_name, role)
            VALUES (?, ?, ?, ?, ?)
        ''', (username.strip(), email.strip().lower(),
              generate_password_hash(password), full_name, role))
        conn.commit()
        return True, "Account created! Please login."
    except sqlite3.IntegrityError as e:
        if 'username' in str(e): return False, "Username already taken."
        if 'email'    in str(e): return False, "Email already registered."
        return False, "Account already exists."
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()


def verify_user(username_or_email, password):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        SELECT * FROM users
        WHERE (username=? OR email=?) AND is_blocked=0
    ''', (username_or_email.strip(), username_or_email.strip().lower()))
    user = c.fetchone()
    conn.close()
    if user and check_password_hash(user['password_hash'], password):
        return dict(user)
    return None


def get_user_by_id(uid):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE id=?", (uid,))
    user = c.fetchone()
    conn.close()
    return user


def update_user_profile(uid, data):
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute('''
            UPDATE users SET
                full_name=?, phone=?, blood_group=?, city=?,
                profile_complete=1
            WHERE id=?
        ''', (data.get('full_name',''), data.get('phone',''),
              data.get('blood_group',''), data.get('city',''), uid))
        conn.commit()
        return True, "Profile updated successfully!"
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()


def get_all_users():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE role='user' ORDER BY created_at DESC")
    users = c.fetchall()
    conn.close()
    return users


def toggle_block_user(uid):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT is_blocked FROM users WHERE id=?", (uid,))
    row = c.fetchone()
    if not row:
        conn.close()
        return False, "User not found."
    new_val = 0 if row['is_blocked'] else 1
    c.execute("UPDATE users SET is_blocked=? WHERE id=?", (new_val, uid))
    conn.commit()
    conn.close()
    return True, "User " + ("blocked." if new_val else "unblocked.")


def delete_user(uid):
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute("DELETE FROM users WHERE id=? AND role!='admin'", (uid,))
        conn.commit()
        if c.rowcount == 0:
            return False, "Cannot delete admin or user not found."
        return True, "User deleted."
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()


def get_user_count():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) as cnt FROM users WHERE role='user'")
    n = c.fetchone()['cnt']
    conn.close()
    return n


def search_users(query):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        SELECT * FROM users WHERE role='user'
        AND (username LIKE ? OR email LIKE ? OR full_name LIKE ?)
        ORDER BY created_at DESC
    ''', (f'%{query}%', f'%{query}%', f'%{query}%'))
    users = c.fetchall()
    conn.close()
    return users


# ══════════════════════════════════════════════════════════
# DONOR FUNCTIONS
# ══════════════════════════════════════════════════════════

def insert_donor(data):
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute('''
            INSERT INTO donors
                (user_id, full_name, age, gender, blood_group,
                 city, state, phone, email, weight, last_donated)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        ''', (data.get('user_id'), data['full_name'], data['age'],
              data['gender'], data['blood_group'], data['city'],
              data.get('state',''), data['phone'],
              data.get('email',''), data['weight'],
              data.get('last_donated') or None))
        conn.commit()
        return True, "Donor registered successfully!"
    except sqlite3.IntegrityError as e:
        if 'phone' in str(e): return False, "Phone already registered."
        if 'email' in str(e): return False, "Email already registered."
        return False, "Donor already exists."
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()


def get_all_donors(search=None, blood_group=None, city=None):
    conn = get_db_connection()
    c = conn.cursor()
    q = "SELECT * FROM donors WHERE 1=1"
    p = []
    if search:
        q += " AND (full_name LIKE ? OR phone LIKE ?)"
        p += [f'%{search}%', f'%{search}%']
    if blood_group:
        q += " AND blood_group=?"
        p.append(blood_group)
    if city:
        q += " AND city LIKE ?"
        p.append(f'%{city}%')
    q += " ORDER BY created_at DESC"
    c.execute(q, p)
    donors = c.fetchall()
    conn.close()
    return donors


def get_donor_by_id(did):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM donors WHERE id=?", (did,))
    d = c.fetchone()
    conn.close()
    return d


def get_donor_by_user_id(uid):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM donors WHERE user_id=?", (uid,))
    d = c.fetchone()
    conn.close()
    return d


def update_donor(did, data):
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute('''
            UPDATE donors SET
                full_name=?, age=?, gender=?, blood_group=?,
                city=?, state=?, phone=?, email=?,
                weight=?, last_donated=?, is_available=?
            WHERE id=?
        ''', (data['full_name'], data['age'], data['gender'],
              data['blood_group'], data['city'], data.get('state',''),
              data['phone'], data.get('email',''), data['weight'],
              data.get('last_donated') or None,
              data.get('is_available', 1), did))
        conn.commit()
        if c.rowcount == 0: return False, "Donor not found."
        return True, "Donor updated!"
    except sqlite3.IntegrityError as e:
        if 'phone' in str(e): return False, "Phone used by another donor."
        if 'email' in str(e): return False, "Email used by another donor."
        return False, "Update failed."
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()


def toggle_donor_availability(did):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT is_available FROM donors WHERE id=?", (did,))
    row = c.fetchone()
    if not row:
        conn.close()
        return False, "Donor not found."
    new_val = 0 if row['is_available'] else 1
    c.execute("UPDATE donors SET is_available=? WHERE id=?", (new_val, did))
    conn.commit()
    conn.close()
    return True, "Donor " + ("activated." if new_val else "deactivated.")


def delete_donor(did):
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute("DELETE FROM donors WHERE id=?", (did,))
        conn.commit()
        if c.rowcount == 0: return False, "Donor not found."
        return True, "Donor deleted."
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()


def get_donor_count():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) as cnt FROM donors")
    n = c.fetchone()['cnt']
    conn.close()
    return n


def get_available_donor_count():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) as cnt FROM donors WHERE is_available=1")
    n = c.fetchone()['cnt']
    conn.close()
    return n


def get_eligible_donor_count():
    conn = get_db_connection()
    c = conn.cursor()
    cutoff = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    c.execute('''
        SELECT COUNT(*) as cnt FROM donors
        WHERE age BETWEEN 18 AND 65
        AND weight >= 45
        AND (last_donated IS NULL OR last_donated <= ?)
        AND is_available = 1
    ''', (cutoff,))
    n = c.fetchone()['cnt']
    conn.close()
    return n


def get_donors_by_blood_group():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        SELECT blood_group, COUNT(*) as cnt
        FROM donors GROUP BY blood_group ORDER BY blood_group
    ''')
    rows = c.fetchall()
    conn.close()
    return {r['blood_group']: r['cnt'] for r in rows}


def get_donors_by_city(limit=10):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        SELECT city, COUNT(*) as cnt FROM donors
        GROUP BY city ORDER BY cnt DESC LIMIT ?
    ''', (limit,))
    rows = c.fetchall()
    conn.close()
    return {r['city']: r['cnt'] for r in rows}


def get_monthly_registrations():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        SELECT strftime('%Y-%m', created_at) as month, COUNT(*) as cnt
        FROM donors
        GROUP BY month ORDER BY month DESC LIMIT 12
    ''')
    rows = c.fetchall()
    conn.close()
    return {r['month']: r['cnt'] for r in reversed(rows)}


def get_recent_donors(limit=5):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM donors ORDER BY created_at DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    return rows


def get_distinct_cities():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT DISTINCT city FROM donors ORDER BY city")
    rows = c.fetchall()
    conn.close()
    return [r['city'] for r in rows]


# ══════════════════════════════════════════════════════════
# EMERGENCY REQUEST FUNCTIONS
# ══════════════════════════════════════════════════════════

def create_emergency_request(data):
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute('''
            INSERT INTO emergency_requests
                (user_id, patient_name, blood_group, hospital,
                 district, contact_phone, units_needed, notes)
            VALUES (?,?,?,?,?,?,?,?)
        ''', (data.get('user_id'), data['patient_name'],
              data['blood_group'], data['hospital'],
              data['district'], data['contact_phone'],
              data.get('units_needed', 1), data.get('notes', '')))
        conn.commit()
        return True, "Emergency request submitted!"
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()


def get_all_emergency_requests():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        SELECT e.*, u.username, u.email as user_email
        FROM emergency_requests e
        LEFT JOIN users u ON e.user_id = u.id
        ORDER BY e.created_at DESC
    ''')
    rows = c.fetchall()
    conn.close()
    return rows


def get_user_emergency_requests(uid):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        SELECT * FROM emergency_requests
        WHERE user_id=? ORDER BY created_at DESC
    ''', (uid,))
    rows = c.fetchall()
    conn.close()
    return rows


def update_emergency_status(req_id, status, admin_notes=''):
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute('''
            UPDATE emergency_requests
            SET status=?, admin_notes=?, updated_at=datetime('now')
            WHERE id=?
        ''', (status, admin_notes, req_id))
        conn.commit()
        return True, f"Request marked as {status}."
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()


def delete_emergency_request(req_id):
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute("DELETE FROM emergency_requests WHERE id=?", (req_id,))
        conn.commit()
        return True, "Request deleted."
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()


def get_emergency_count_by_status(status):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) as cnt FROM emergency_requests WHERE status=?", (status,))
    n = c.fetchone()['cnt']
    conn.close()
    return n


# ══════════════════════════════════════════════════════════
# ANNOUNCEMENTS
# ══════════════════════════════════════════════════════════

def get_active_announcements():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        SELECT * FROM announcements WHERE is_active=1
        ORDER BY created_at DESC LIMIT 5
    ''')
    rows = c.fetchall()
    conn.close()
    return rows


def create_announcement(title, content):
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute('''
            INSERT INTO announcements (title, content) VALUES (?,?)
        ''', (title, content))
        conn.commit()
        return True, "Announcement created."
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()


def delete_announcement(ann_id):
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute("DELETE FROM announcements WHERE id=?", (ann_id,))
        conn.commit()
        return True, "Announcement deleted."
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()


if __name__ == '__main__':
    create_tables()