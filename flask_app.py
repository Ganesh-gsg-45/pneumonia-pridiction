# CHANGES TO flask_app.py
# ========================
# Builds on the app's EXISTING Postgres auth (SQLAlchemy User model,
# session-based login) — adds: (1) a gate requiring login, (2) Google OAuth
# using the same session + same User table, just with two new nullable
# columns.

# ---------------------------------------------------------------------------
# 1. New imports — add near your existing imports
# ---------------------------------------------------------------------------
from functools import wraps
from authlib.integrations.flask_client import OAuth

# ---------------------------------------------------------------------------
# 2. Update the User model — add google_id, make password_hash nullable
#    (Google-only accounts won't have a password)
# ---------------------------------------------------------------------------
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=True)   # CHANGED: nullable=True
    google_id = db.Column(db.String(255), unique=True, nullable=True)  # NEW
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None
        }

# NOTE: if you already have a database with the OLD schema (no google_id
# column), db.create_all() will NOT add the new column to an existing table
# — it only creates missing tables. Either:
#   (a) drop the users table and let it recreate fresh (fine if you have no
#       real users yet — you're mid-development), or
#   (b) run this manually in pgAdmin against your local db:
#       ALTER TABLE users ADD COLUMN google_id VARCHAR(255) UNIQUE;
#       ALTER TABLE users ALTER COLUMN password_hash DROP NOT NULL;


# ---------------------------------------------------------------------------
# 3. login_required decorator — add after the User model / db.create_all()
#    block, before any @app.route definitions
# ---------------------------------------------------------------------------
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get('user_id'):
            if request.path.startswith('/api/') or request.path in ('/analyze', '/predict', '/chat'):
                return jsonify({'error': 'Please log in to continue.'}), 401
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# 4. Google OAuth setup — add right after app.secret_key is set
# ---------------------------------------------------------------------------
oauth = OAuth(app)
oauth.register(
    name="google",
    client_id=os.environ.get("GOOGLE_CLIENT_ID"),
    client_secret=os.environ.get("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)


# ---------------------------------------------------------------------------
# 5. New routes — add near your existing /api/login etc.
# ---------------------------------------------------------------------------
@app.route('/login')
def login_page():
    if session.get('user_id'):
        return redirect(url_for('index'))
    return render_template('login.html')


@app.route('/signup')
def signup_page():
    if session.get('user_id'):
        return redirect(url_for('index'))
    return render_template('signup.html')


@app.route('/login/google')
def google_login():
    redirect_uri = url_for('google_callback', _external=True)
    return oauth.google.authorize_redirect(redirect_uri)


@app.route('/login/google/callback')
def google_callback():
    token = oauth.google.authorize_access_token()
    userinfo = token.get('userinfo')
    if not userinfo:
        return redirect(url_for('login_page'))

    google_id = userinfo['sub']
    email = userinfo['email'].lower()
    name = userinfo.get('name', '') or email.split('@')[0]

    user = User.query.filter_by(google_id=google_id).first()

    if not user:
        # Not found by google_id — check if this email already has a
        # password-based account, and link Google to it instead of
        # creating a duplicate.
        user = User.query.filter_by(email=email).first()
        if user:
            user.google_id = google_id
            db.session.commit()
        else:
            # New account via Google. Generate a unique username from the
            # email prefix (usernames must be unique in this schema).
            base_username = name.replace(' ', '').lower()[:70] or email.split('@')[0]
            username = base_username
            suffix = 1
            while User.query.filter_by(username=username).first():
                suffix += 1
                username = f"{base_username}{suffix}"

            user = User(username=username, email=email, google_id=google_id, password_hash=None)
            db.session.add(user)
            db.session.commit()

    session['user_id'] = user.id
    session['username'] = user.username
    return redirect(url_for('index'))


# ---------------------------------------------------------------------------
# 6. Add @login_required to these existing routes (add the line directly
#    above each @app.route — nothing else in the function bodies changes)
# ---------------------------------------------------------------------------

@app.route('/')
@login_required                     # <-- ADD THIS LINE
def index():
    user_id = session.get('user_id')
    user = db.session.get(User, user_id) if user_id else None
    return render_template('index.html', ai_connected=OPENAI_AVAILABLE, current_user=user.to_dict() if user else None)


@app.route('/analyze', methods=['POST'])
@login_required                     # <-- ADD THIS LINE
def analyze():
    ...  # unchanged


@app.route('/predict', methods=['POST'])
@login_required                     # <-- ADD THIS LINE
def predict_alias():
    return analyze()


@app.route('/chat', methods=['POST'])
@login_required                     # <-- ADD THIS LINE
def chat():
    ...  # unchanged


# NOTE: /api/login, /api/signup, /api/logout, /api/me, /login/google,
# /login/google/callback all stay WITHOUT @login_required — they need to be
# reachable by people who aren't logged in yet.