import re
import csv
import io
from datetime import datetime, timedelta
from functools import wraps
from flask import (Flask, render_template, request, redirect,
                   url_for, flash, session, Response)
from config import Config
from database import *
# ── ML Predictor ──────────────────────────────────────────
try:
    from ml.predictor import (
        predict_eligibility,
        get_ai_recommendations,
        forecast_blood_demand,
        get_model_status
    )
    ML_AVAILABLE = True
    print("✅ ML models loaded successfully.")
except ImportError as e:
    ML_AVAILABLE = False
    print(f"⚠️ ML not available: {e}")

app = Flask(__name__)
app.config.from_object(Config)

with app.app_context():
    create_tables()

# ── Constants ─────────────────────────────────────────────
BLOOD_GROUPS = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']

TAMIL_NADU_DISTRICTS = [
    "Ariyalur", "Chengalpattu", "Chennai", "Coimbatore",
    "Cuddalore", "Dharmapuri", "Dindigul", "Erode",
    "Kallakurichi", "Kancheepuram", "Kanniyakumari",
    "Karur", "Krishnagiri", "Madurai", "Mayiladuthurai",
    "Nagapattinam", "Namakkal", "Nilgiris", "Perambalur",
    "Pudukkottai", "Ramanathapuram", "Ranipet", "Salem",
    "Sivaganga", "Tenkasi", "Thanjavur", "Theni",
    "Thoothukudi", "Tiruchirappalli", "Tirunelveli",
    "Tirupathur", "Tiruppur", "Tiruvallur", "Tiruvannamalai",
    "Tiruvarur", "Vellore", "Viluppuram", "Virudhunagar"
]


# ══════════════════════════════════════════════════════════
# ACCESS CONTROL DECORATORS
# ══════════════════════════════════════════════════════════

def login_required(f):
    """Any logged-in user (user or admin)."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash("Please login to continue.", "danger")
            return redirect(url_for('user_login'))
        if session.get('is_blocked'):
            flash("Your account has been blocked. Contact support.", "danger")
            session.clear()
            return redirect(url_for('user_login'))
        return f(*args, **kwargs)
    return decorated


def user_only(f):
    """Only normal users (not admin)."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash("Please login to continue.", "danger")
            return redirect(url_for('user_login'))
        if session.get('role') == 'admin':
            return redirect(url_for('admin_dashboard'))
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    """Only admins."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash("Admin login required.", "danger")
            return redirect(url_for('admin_login'))
        if session.get('role') != 'admin':
            flash("⚠️ Access Denied! This page is for Admins only.", "danger")
            return redirect(url_for('home'))
        return f(*args, **kwargs)
    return decorated


# ── Validation ────────────────────────────────────────────
def validate_donor_form(form):
    errors = {}
    full_name   = form.get('full_name','').strip()
    age         = form.get('age','').strip()
    gender      = form.get('gender','').strip()
    blood_group = form.get('blood_group','').strip()
    city        = form.get('city','').strip()
    phone       = form.get('phone','').strip()
    weight      = form.get('weight','').strip()

    if not full_name:   errors['full_name']   = "Full name is required."
    if not age:         errors['age']         = "Age is required."
    if not gender:      errors['gender']      = "Gender is required."
    if not blood_group: errors['blood_group'] = "Blood group is required."
    if not city:        errors['city']        = "City is required."
    if not phone:       errors['phone']       = "Phone is required."
    if not weight:      errors['weight']      = "Weight is required."

    if age:
        if not age.isdigit():
            errors['age'] = "Enter a valid age."
        elif not (18 <= int(age) <= 65):
            errors['age'] = "Age must be between 18 and 65."

    if weight:
        try:
            if float(weight) < 45:
                errors['weight'] = "Weight must be at least 45 kg."
        except ValueError:
            errors['weight'] = "Enter a valid weight."

    if phone and not re.match(r'^(\+91[\-\s]?)?[6-9]\d{9}$', phone):
        errors['phone'] = "Enter a valid 10-digit phone number."

    email = form.get('email','').strip()
    if email and not re.match(r'^[\w\.\-]+@[\w\-]+\.[a-zA-Z]{2,}$', email):
        errors['email'] = "Enter a valid email address."

    if blood_group and blood_group not in BLOOD_GROUPS:
        errors['blood_group'] = "Select a valid blood group."

    if gender and gender not in ['Male','Female','Other']:
        errors['gender'] = "Select a valid gender."

    last_donated = form.get('last_donated','').strip()
    if last_donated:
        try:
            if datetime.strptime(last_donated,'%Y-%m-%d').date() > datetime.now().date():
                errors['last_donated'] = "Last donation date cannot be in the future."
        except ValueError:
            errors['last_donated'] = "Enter a valid date."

    return errors


# ══════════════════════════════════════════════════════════
# PUBLIC ROUTES
# ══════════════════════════════════════════════════════════

@app.route('/')
def home():
    announcements = get_active_announcements()
    stats = {
        'total_donors':   get_donor_count(),
        'cities_covered': len(get_distinct_cities()),
        'lives_saved':    get_donor_count() * 3,
        'blood_groups':   8,
    }
    return render_template('index.html',
        stats=stats, announcements=announcements)


@app.route('/privacy')
def privacy():
    return render_template('privacy.html')


# ══════════════════════════════════════════════════════════
# USER AUTH ROUTES
# ══════════════════════════════════════════════════════════

@app.route('/login', methods=['GET','POST'])
def user_login():
    if 'user_id' in session:
        if session.get('role') == 'admin':
            return redirect(url_for('admin_dashboard'))
        return redirect(url_for('home'))

    form_data = {}
    errors    = {}

    if request.method == 'POST':
        form_data = request.form.to_dict()
        username  = form_data.get('username','').strip()
        password  = form_data.get('password','').strip()

        if not username: errors['username'] = "Username or email is required."
        if not password: errors['password'] = "Password is required."

        if not errors:
            user = verify_user(username, password)
            if user:
                if user['role'] == 'admin':
                    flash("Please use the Admin Login page.", "warning")
                    return redirect(url_for('admin_login'))
                session['user_id']    = user['id']
                session['username']   = user['username']
                session['role']       = user['role']
                session['full_name']  = user['full_name'] or user['username']
                session['is_blocked'] = user['is_blocked']
                flash(f"Welcome back, {user['username']}! 🩸", 'success')
                return redirect(url_for('home'))
            else:
                flash("Invalid credentials or account blocked.", 'danger')

    return render_template('auth/login.html',
        form_data=form_data, errors=errors)


@app.route('/signup', methods=['GET','POST'])
def user_signup():
    if 'user_id' in session:
        return redirect(url_for('home'))

    form_data = {}
    errors    = {}

    if request.method == 'POST':
        form_data  = request.form.to_dict()
        username   = form_data.get('username','').strip()
        email      = form_data.get('email','').strip()
        full_name  = form_data.get('full_name','').strip()
        password   = form_data.get('password','').strip()
        confirm    = form_data.get('confirm_password','').strip()

        if not full_name: errors['full_name'] = "Full name is required."
        if not username or len(username) < 3:
            errors['username'] = "Username must be at least 3 characters."
        if not email or not re.match(r'^[\w\.\-]+@[\w\-]+\.[a-zA-Z]{2,}$', email):
            errors['email'] = "Enter a valid email."
        if not password or len(password) < 6:
            errors['password'] = "Password must be at least 6 characters."
        if password != confirm:
            errors['confirm_password'] = "Passwords do not match."

        if not errors:
            success, message = create_user(username, email, password, full_name)
            if success:
                flash(message, 'success')
                return redirect(url_for('user_login'))
            else:
                flash(message, 'danger')

    return render_template('auth/signup.html',
        form_data=form_data, errors=errors)


@app.route('/logout')
def user_logout():
    name = session.get('username', 'User')
    session.clear()
    flash(f"Goodbye, {name}! You have been logged out.", 'success')
    return redirect(url_for('home'))


# ══════════════════════════════════════════════════════════
# USER ROUTES (login required)
# ══════════════════════════════════════════════════════════

@app.route('/profile', methods=['GET','POST'])
@login_required
def user_profile():
    user  = get_user_by_id(session['user_id'])
    donor = get_donor_by_user_id(session['user_id'])
    my_requests = get_user_emergency_requests(session['user_id'])

    if request.method == 'POST':
        data = request.form.to_dict()
        success, message = update_user_profile(session['user_id'], data)
        flash(message, 'success' if success else 'danger')
        session['full_name'] = data.get('full_name', session.get('full_name'))
        return redirect(url_for('user_profile'))

    return render_template('profile.html',
        user=user, donor=donor,
        my_requests=my_requests,
        blood_groups=BLOOD_GROUPS)


@app.route('/register', methods=['GET','POST'])
@login_required
def register():
    existing_donor = get_donor_by_user_id(session['user_id'])
    form_data = {}

    if request.method == 'POST':
        form_data = request.form.to_dict()
        errors    = validate_donor_form(request.form)

        if errors:
            return render_template('register.html',
        blood_groups=BLOOD_GROUPS, errors={}, form_data=form_data,
        existing_donor=existing_donor,
        stats={'total_donors': get_donor_count()})

        donor_data = {
            'user_id':      session['user_id'],
            'full_name':    form_data['full_name'].strip(),
            'age':          int(form_data['age']),
            'gender':       form_data['gender'],
            'blood_group':  form_data['blood_group'],
            'city':         form_data['city'].strip(),
            'state':        form_data.get('state','').strip(),
            'phone':        form_data['phone'].strip(),
            'email':        form_data.get('email','').strip(),
            'weight':       float(form_data['weight']),
            'last_donated': form_data.get('last_donated','').strip(),
        }
        success, message = insert_donor(donor_data)
        flash(message, 'success' if success else 'danger')
        if success:
            return redirect(url_for('user_profile'))

    return render_template('register.html',
        blood_groups=BLOOD_GROUPS, errors={}, form_data=form_data,
        existing_donor=existing_donor,
        stats={'total_donors': get_donor_count()})


@app.route('/search', methods=['GET','POST'])
@login_required
def search_donor():
    results     = []
    searched    = False
    blood_group = ''

    if request.method == 'POST':
        blood_group = request.form.get('blood_group','').strip()
        searched    = True
        results     = get_all_donors(
            blood_group=blood_group if blood_group else None)

    return render_template('search.html',
        blood_groups=BLOOD_GROUPS, results=results,
        searched=searched, selected_blood_group=blood_group,
        total_results=len(results))


@app.route('/location-search', methods=['GET','POST'])
@login_required
def location_search():
    results          = []
    searched         = False
    blood_group      = ''
    district         = ''
    ai_recommendations = []

    if request.method == 'POST':
        blood_group = request.form.get('blood_group','').strip()
        district    = request.form.get('district','').strip()
        searched    = True

        results = get_all_donors(
            blood_group=blood_group if blood_group else None,
            city=district if district else None)

        # ── AI Recommendations ─────────────────────────────
        if ML_AVAILABLE and results:
            try:
                ai_recommendations = get_ai_recommendations(
                    donors=results,
                    target_blood_group=blood_group or 'O+',
                    target_district=district or '',
                    emergency_level=2,
                    top_n=5
                )
            except Exception as e:
                print(f"⚠️ Recommendation error: {e}")
                ai_recommendations = []

    return render_template('location_search.html',
        blood_groups=BLOOD_GROUPS,
        districts=TAMIL_NADU_DISTRICTS,
        results=results,
        searched=searched,
        selected_blood_group=blood_group,
        selected_district=district,
        total_results=len(results),
        ai_recommendations=ai_recommendations,
        ml_available=ML_AVAILABLE)


@app.route('/eligibility', methods=['GET','POST'])
@login_required
def eligibility():
    result    = None
    form_data = {}

    if request.method == 'POST':
        form_data    = request.form.to_dict()
        age_str      = form_data.get('age','').strip()
        weight_str   = form_data.get('weight','').strip()
        last_donated = form_data.get('last_donated','').strip()

        # Basic validation before ML prediction
        errors = []
        age    = None
        weight = None

        try:
            age = int(age_str)
        except ValueError:
            errors.append("Please enter a valid age.")

        try:
            weight = float(weight_str)
        except ValueError:
            errors.append("Please enter a valid weight.")

        if last_donated:
            try:
                last_date = datetime.strptime(last_donated, '%Y-%m-%d').date()
                if last_date > datetime.now().date():
                    errors.append("Last donation date cannot be in the future.")
                    last_donated = None
            except ValueError:
                errors.append("Enter a valid last donation date.")
                last_donated = None

        if errors:
            result = {
                'is_eligible':            False,
                'confidence':             0,
                'probability_eligible':   0,
                'probability_ineligible': 100,
                'reasons':                [f"✘ {e}" for e in errors],
                'model_used':             'Validation Error',
                'days_since_donation':    None
            }
        elif ML_AVAILABLE:
            # ── AI/ML Prediction ──────────────────────────
            result = predict_eligibility(
                age=age,
                weight=weight,
                last_donated_date=last_donated if last_donated else None
            )
        else:
            # ── Fallback rule-based ────────────────────────
            from ml.predictor import _rule_based_eligibility
            has_donated = 1 if last_donated else 0
            days_since  = 0
            if last_donated:
                try:
                    last_date  = datetime.strptime(
                        last_donated, '%Y-%m-%d').date()
                    days_since = (datetime.now().date() - last_date).days
                except ValueError:
                    pass
            result = _rule_based_eligibility(
                age, weight, has_donated, days_since)

    return render_template('eligibility.html',
        result=result,
        form_data=form_data,
        ml_available=ML_AVAILABLE)


@app.route('/emergency-request', methods=['GET','POST'])
@login_required
def emergency_request():
    my_requests = get_user_emergency_requests(session['user_id'])

    if request.method == 'POST':
        data = {
            'user_id':       session['user_id'],
            'patient_name':  request.form.get('patient_name','').strip(),
            'blood_group':   request.form.get('blood_group','').strip(),
            'hospital':      request.form.get('hospital','').strip(),
            'district':      request.form.get('district','').strip(),
            'contact_phone': request.form.get('contact_phone','').strip(),
            'units_needed':  request.form.get('units_needed', 1),
            'notes':         request.form.get('notes','').strip(),
        }

        if not all([data['patient_name'], data['blood_group'],
                    data['hospital'], data['district'],
                    data['contact_phone']]):
            flash("All required fields must be filled.", 'danger')
        else:
            success, message = create_emergency_request(data)
            flash(message, 'success' if success else 'danger')
            if success:
                return redirect(url_for('emergency_request'))

    return render_template('emergency_request.html',
        blood_groups=BLOOD_GROUPS,
        districts=TAMIL_NADU_DISTRICTS,
        my_requests=my_requests)


# ══════════════════════════════════════════════════════════
# ADMIN AUTH ROUTES
# ══════════════════════════════════════════════════════════

@app.route('/admin/login', methods=['GET','POST'])
def admin_login():
    if session.get('role') == 'admin':
        return redirect(url_for('admin_dashboard'))

    form_data = {}

    if request.method == 'POST':
        form_data = request.form.to_dict()
        username  = form_data.get('username','').strip()
        password  = form_data.get('password','').strip()

        if not username or not password:
            flash("Username and password are required.", 'danger')
        else:
            user = verify_user(username, password)
            if user and user['role'] == 'admin':
                session['user_id']   = user['id']
                session['username']  = user['username']
                session['role']      = user['role']
                session['full_name'] = user['full_name'] or 'Admin'
                flash(f"Welcome, Admin {user['username']}! 🛡️", 'success')
                return redirect(url_for('admin_dashboard'))
            else:
                flash("Invalid admin credentials.", 'danger')

    return render_template('auth/admin_login.html', form_data=form_data)


@app.route('/admin/logout')
def admin_logout():
    session.clear()
    flash("Admin logged out successfully.", 'success')
    return redirect(url_for('admin_login'))


# ══════════════════════════════════════════════════════════
# ADMIN ROUTES
# ══════════════════════════════════════════════════════════

@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    total_donors    = get_donor_count()
    eligible_donors = get_eligible_donor_count()
    total_users     = get_user_count()
    total_cities    = len(get_distinct_cities())
    blood_group_data = get_donors_by_blood_group()
    city_data        = get_donors_by_city()
    recent_donors    = get_recent_donors(5)
    all_reqs         = get_all_emergency_requests()
    pending_reqs     = get_emergency_count_by_status('pending')
    approved_reqs    = get_emergency_count_by_status('approved')

    full_bg_data   = {bg: blood_group_data.get(bg,0) for bg in BLOOD_GROUPS}
    ineligible     = total_donors - eligible_donors
    eligible_pct   = round(eligible_donors/total_donors*100, 1) if total_donors > 0 else 0

    return render_template('admin/dashboard.html',
        total_donors=total_donors,
        eligible_donors=eligible_donors,
        ineligible_donors=ineligible,
        eligible_pct=eligible_pct,
        ineligible_pct=round(100-eligible_pct, 1),
        total_users=total_users,
        total_cities=total_cities,
        available_donors=get_available_donor_count(),
        blood_group_data=full_bg_data,
        city_data=city_data,
        recent_donors=recent_donors,
        pending_reqs=pending_reqs,
        approved_reqs=approved_reqs,
        now=datetime.now())


# ── Admin: Manage Donors ──────────────────────────────────

@app.route('/admin/donors')
@admin_required
def admin_donors():
    page        = request.args.get('page', 1, type=int)
    per_page    = 10
    search      = request.args.get('search','').strip()
    blood_group = request.args.get('blood_group','').strip()
    sort_by     = request.args.get('sort','created_at')
    order       = request.args.get('order','desc')

    all_donors = get_all_donors(
        search=search if search else None,
        blood_group=blood_group if blood_group else None)

    sort_map = {
        'name':'full_name','blood_group':'blood_group',
        'city':'city','age':'age','created_at':'created_at'
    }
    sort_key = sort_map.get(sort_by,'created_at')
    try:
        all_donors = sorted(all_donors,
            key=lambda x: x[sort_key] or '', reverse=(order=='desc'))
    except Exception:
        pass

    total       = len(all_donors)
    total_pages = max(1,(total+per_page-1)//per_page)
    page        = max(1, min(page, total_pages))
    start       = (page-1)*per_page
    paginated   = all_donors[start:start+per_page]

    return render_template('admin/manage_donors.html',
        donors=paginated,
        blood_groups=BLOOD_GROUPS,
        total=total, page=page,
        per_page=per_page,
        total_pages=total_pages,
        search=search,
        selected_blood_group=blood_group,
        sort_by=sort_by, order=order)


@app.route('/admin/donors/update', methods=['GET','POST'])
@admin_required
def admin_update_donor():
    donor_id  = request.args.get('id','') or request.form.get('donor_id','')
    donor     = None
    errors    = {}
    form_data = {}

    if request.method == 'POST' and request.form.get('action') == 'save':
        donor_id  = request.form.get('donor_id','')
        form_data = request.form.to_dict()
        errors    = validate_donor_form(request.form)

        if not errors:
            donor_data = {
                'full_name':    form_data['full_name'].strip(),
                'age':          int(form_data['age']),
                'gender':       form_data['gender'],
                'blood_group':  form_data['blood_group'],
                'city':         form_data['city'].strip(),
                'state':        form_data.get('state','').strip(),
                'phone':        form_data['phone'].strip(),
                'email':        form_data.get('email','').strip(),
                'weight':       float(form_data['weight']),
                'last_donated': form_data.get('last_donated','').strip(),
                'is_available': 1 if form_data.get('is_available')=='on' else 0,
            }
            success, message = update_donor(int(donor_id), donor_data)
            flash(message, 'success' if success else 'danger')
            if success:
                return redirect(url_for('admin_donors'))
        donor = get_donor_by_id(int(donor_id))

    elif donor_id:
        try:
            donor = get_donor_by_id(int(donor_id))
            if not donor:
                flash(f"No donor with ID {donor_id}.", 'danger')
        except ValueError:
            flash("Invalid ID.", 'danger')

    return render_template('admin/update_donor.html',
        donor=donor, donor_id=donor_id,
        blood_groups=BLOOD_GROUPS,
        errors=errors, form_data=form_data)


@app.route('/admin/donors/toggle/<int:did>')
@admin_required
def admin_toggle_donor(did):
    success, message = toggle_donor_availability(did)
    flash(message, 'success' if success else 'danger')
    return redirect(url_for('admin_donors'))


@app.route('/admin/donors/delete/<int:did>', methods=['POST'])
@admin_required
def admin_delete_donor(did):
    success, message = delete_donor(did)
    flash(message, 'success' if success else 'danger')
    return redirect(url_for('admin_donors'))


# ── Admin: Manage Users ───────────────────────────────────

@app.route('/admin/users')
@admin_required
def admin_users():
    search = request.args.get('search','').strip()
    if search:
        users = search_users(search)
    else:
        users = get_all_users()
    return render_template('admin/manage_users.html',
        users=users, search=search)


@app.route('/admin/users/block/<int:uid>')
@admin_required
def admin_block_user(uid):
    success, message = toggle_block_user(uid)
    flash(message, 'success' if success else 'danger')
    return redirect(url_for('admin_users'))


@app.route('/admin/users/delete/<int:uid>', methods=['POST'])
@admin_required
def admin_delete_user(uid):
    success, message = delete_user(uid)
    flash(message, 'success' if success else 'danger')
    return redirect(url_for('admin_users'))


# ── Admin: Emergency Requests ─────────────────────────────

@app.route('/admin/emergency')
@admin_required
def admin_emergency():
    status_filter = request.args.get('status','').strip()
    all_reqs = get_all_emergency_requests()
    if status_filter:
        all_reqs = [r for r in all_reqs if r['status'] == status_filter]
    return render_template('admin/emergency.html',
        requests=all_reqs,
        status_filter=status_filter,
        pending=get_emergency_count_by_status('pending'),
        approved=get_emergency_count_by_status('approved'),
        completed=get_emergency_count_by_status('completed'),
        rejected=get_emergency_count_by_status('rejected'))


@app.route('/admin/emergency/update/<int:req_id>/<status>')
@admin_required
def admin_update_emergency(req_id, status):
    valid = ['pending','approved','completed','rejected']
    if status not in valid:
        flash("Invalid status.", 'danger')
        return redirect(url_for('admin_emergency'))
    admin_notes = request.args.get('notes','')
    success, message = update_emergency_status(req_id, status, admin_notes)
    flash(message, 'success' if success else 'danger')
    return redirect(url_for('admin_emergency'))


@app.route('/admin/emergency/delete/<int:req_id>', methods=['POST'])
@admin_required
def admin_delete_emergency(req_id):
    success, message = delete_emergency_request(req_id)
    flash(message, 'success' if success else 'danger')
    return redirect(url_for('admin_emergency'))


# ── Admin: Analytics ──────────────────────────────────────

@app.route('/admin/analytics')
@admin_required
def admin_analytics():
    total_donors     = get_donor_count()
    eligible_donors  = get_eligible_donor_count()
    available_donors = get_available_donor_count()
    blood_group_data = get_donors_by_blood_group()
    city_data        = get_donors_by_city()
    monthly_data     = get_monthly_registrations()
    full_bg_data     = {bg: blood_group_data.get(bg,0) for bg in BLOOD_GROUPS}
    ineligible       = total_donors - eligible_donors
    eligible_pct     = round(eligible_donors/total_donors*100, 1) if total_donors > 0 else 0

    # ── ML Blood Demand Forecasting ────────────────────────
    forecast = None
    if ML_AVAILABLE:
        try:
            forecast = forecast_blood_demand(
                blood_groups=BLOOD_GROUPS,
                districts=list(city_data.keys()) or TAMIL_NADU_DISTRICTS[:10],
                registrations=max(total_donors, 10),
                emergency_requests=5
            )
        except Exception as e:
            print(f"⚠️ Forecast error: {e}")

    return render_template('admin/analytics.html',
        total_donors=total_donors,
        eligible_donors=eligible_donors,
        ineligible_donors=ineligible,
        eligible_pct=eligible_pct,
        ineligible_pct=round(100-eligible_pct, 1),
        blood_group_data=full_bg_data,
        city_data=city_data,
        monthly_data=monthly_data,
        available_donors=available_donors,
        forecast=forecast,
        ml_available=ML_AVAILABLE)


# ── Admin: Announcements ──────────────────────────────────

@app.route('/admin/announcements', methods=['GET','POST'])
@admin_required
def admin_announcements():
    from database import get_active_announcements, delete_announcement
    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'create':
            title   = request.form.get('title','').strip()
            content = request.form.get('content','').strip()
            if title and content:
                success, msg = create_announcement(title, content)
                flash(msg, 'success' if success else 'danger')
            else:
                flash("Title and content are required.", 'danger')
        elif action == 'delete':
            ann_id = request.form.get('ann_id')
            if ann_id:
                success, msg = delete_announcement(int(ann_id))
                flash(msg, 'success' if success else 'danger')
        return redirect(url_for('admin_announcements'))

    announcements = get_active_announcements()
    return render_template('admin/announcements.html',
        announcements=announcements)


# ── Admin: Export CSV ─────────────────────────────────────

@app.route('/admin/export/donors')
@admin_required
def admin_export_donors():
    donors = get_all_donors()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID','Name','Blood Group','Age','Gender',
                     'Weight','Phone','Email','City','State',
                     'Available','Last Donated','Registered On'])
    for d in donors:
        writer.writerow([
            d['id'], d['full_name'], d['blood_group'], d['age'],
            d['gender'], d['weight'], d['phone'], d['email'] or '',
            d['city'], d['state'] or '',
            'Yes' if d['is_available'] else 'No',
            d['last_donated'] or 'N/A', d['created_at']
        ])
    output.seek(0)
    filename = f"lifelink_donors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(output.getvalue(), mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename={filename}'})


@app.route('/admin/export/emergency')
@admin_required
def admin_export_emergency():
    reqs   = get_all_emergency_requests()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID','Patient','Blood Group','Hospital',
                     'District','Phone','Units','Status',
                     'Requested By','Date'])
    for r in reqs:
        writer.writerow([
            r['id'], r['patient_name'], r['blood_group'],
            r['hospital'], r['district'], r['contact_phone'],
            r['units_needed'], r['status'],
            r['username'] or 'N/A', r['created_at']
        ])
    output.seek(0)
    filename = f"lifelink_emergency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(output.getvalue(), mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename={filename}'})

# ── Admin: Donor Filter API (for clickable analytics cards) ──

@app.route('/admin/donors/filter')
@admin_required
def admin_donors_filter():
    """Filter donors by type for analytics card clicks."""
    filter_type = request.args.get('type', 'all').strip()
    page        = request.args.get('page', 1, type=int)
    per_page    = 15

    all_donors = get_all_donors()

    # Apply filter
    from datetime import timedelta
    cutoff = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')

    if filter_type == 'eligible':
        filtered = [d for d in all_donors
                    if d['age'] >= 18 and d['age'] <= 65
                    and d['weight'] >= 45
                    and d['is_available'] == 1
                    and (d['last_donated'] is None
                         or d['last_donated'] <= cutoff)]
        title = "Eligible Donors"

    elif filter_type == 'ineligible':
        filtered = [d for d in all_donors
                    if not (d['age'] >= 18 and d['age'] <= 65
                            and d['weight'] >= 45
                            and (d['last_donated'] is None
                                 or d['last_donated'] <= cutoff))]
        title = "Ineligible Donors"

    elif filter_type == 'available':
        filtered = [d for d in all_donors if d['is_available'] == 1]
        title    = "Available Donors"

    else:
        filtered = all_donors
        title    = "All Donors"

    total       = len(filtered)
    total_pages = max(1, (total + per_page - 1) // per_page)
    page        = max(1, min(page, total_pages))
    start       = (page - 1) * per_page
    paginated   = filtered[start:start + per_page]

    return render_template('admin/donors_filter.html',
        donors=paginated,
        title=title,
        filter_type=filter_type,
        total=total,
        page=page,
        per_page=per_page,
        total_pages=total_pages)
# ── Run ───────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True)