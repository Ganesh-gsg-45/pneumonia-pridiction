import { initializeApp } from "https://www.gstatic.com/firebasejs/10.8.0/firebase-app.js";
import { 
  getAuth, 
  createUserWithEmailAndPassword, 
  signInWithEmailAndPassword, 
  signInWithPopup,
  signInWithRedirect,
  getRedirectResult,
  GoogleAuthProvider,
  signOut, 
  onAuthStateChanged 
} from "https://www.gstatic.com/firebasejs/10.8.0/firebase-auth.js";

const firebaseConfig = {
  apiKey: "AIzaSyBMjnGiT5CklJ8BzeQTDV4Ji6yVqpFH11k",
  authDomain: "pneumovision-3734d.firebaseapp.com",
  projectId: "pneumovision-3734d",
  storageBucket: "pneumovision-3734d.firebasestorage.app",
  messagingSenderId: "976973045640",
  appId: "1:976973045640:web:3ccf7611f40ae71720e43d",
  measurementId: "G-TZ7DS76CR7"
};

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const googleProvider = new GoogleAuthProvider();


document.addEventListener('DOMContentLoaded', async () => {
  // Helper to show errors
  const showError = (message) => {
    const errorDiv = document.getElementById('authError');
    if (errorDiv) {
      errorDiv.textContent = message;
      errorDiv.classList.remove('hidden');
      errorDiv.setAttribute('aria-hidden', 'false');
    }
  };

  const hideError = () => {
    const errorDiv = document.getElementById('authError');
    if (errorDiv) {
      errorDiv.textContent = '';
      errorDiv.classList.add('hidden');
      errorDiv.setAttribute('aria-hidden', 'true');
    }
  };

  const setButtonState = (button, busy) => {
    if (!button) return;
    button.disabled = busy;
    button.setAttribute('aria-busy', busy ? 'true' : 'false');
  };

  const isEmailValid = (email) => {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.trim());
  };

  // Handle Login
  const loginForm = document.getElementById('loginForm');
  if (loginForm) {
    loginForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const email = document.getElementById('email').value;
      const password = document.getElementById('password').value;
      const btn = document.getElementById('loginBtn');
      const txt = document.getElementById('loginBtnText');
      const spin = document.getElementById('loginSpinner');

      if (!isEmailValid(email)) {
        showError('Please enter a valid email address.');
        return;
      }

      if (!password) {
        showError('Please enter your password.');
        return;
      }

      setButtonState(btn, true);
      txt.classList.add('hidden');
      spin.classList.remove('hidden');
      hideError();

      try {
        await signInWithEmailAndPassword(auth, email, password);
        window.location.href = '/'; // Redirect to main app
      } catch (error) {
        showError("Login failed: " + error.message.replace('Firebase: ', ''));
        setButtonState(btn, false);
        txt.classList.remove('hidden');
        spin.classList.add('hidden');
      }
    });
  }

  // Handle Signup
  const signupForm = document.getElementById('signupForm');
  if (signupForm) {
    signupForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const email = document.getElementById('email').value;
      const password = document.getElementById('password').value;
      const confirmPassword = document.getElementById('confirmPassword').value;
      const btn = document.getElementById('signupBtn');
      const txt = document.getElementById('signupBtnText');
      const spin = document.getElementById('signupSpinner');

      if (!isEmailValid(email)) {
        showError('Please enter a valid email address.');
        return;
      }

      if (!password) {
        showError('Please enter a password.');
        return;
      }

      if (password !== confirmPassword) {
        showError("Passwords do not match");
        return;
      }

      setButtonState(btn, true);
      txt.classList.add('hidden');
      spin.classList.remove('hidden');
      hideError();

      try {
        await createUserWithEmailAndPassword(auth, email, password);
        window.location.href = '/'; // Redirect to main app
      } catch (error) {
        showError("Signup failed: " + error.message.replace('Firebase: ', ''));
        setButtonState(btn, false);
        txt.classList.remove('hidden');
        spin.classList.add('hidden');
      }
    });
  }

  // Handle Google Login/Signup
  const handleGoogleAuth = async () => {
    try {
      await signInWithPopup(auth, googleProvider);
      window.location.href = '/';
    } catch (error) {
      console.error('Google auth error', error);
      // Popup blocked fallback
      if (error.code === 'auth/popup-blocked' || error.code === 'auth/cancelled-popup-request') {
        try {
          await signInWithRedirect(auth, googleProvider);
        } catch (redirectError) {
          console.error('Google redirect error', redirectError);
          showError("Google redirect sign-in failed: " + redirectError.message.replace('Firebase: ', ''));
        }
      } else if (error.code === 'auth/unauthorized-domain') {
        showError("Google sign-in is blocked because this domain is not authorized in Firebase. Add your Hugging Face Space domain to Firebase Auth authorized domains.");
      } else {
        showError("Google sign-in failed: " + error.code + " — " + error.message.replace('Firebase: ', ''));
      }
    }
  };

  const googleLoginBtn = document.getElementById('googleLoginBtn');
  if (googleLoginBtn) googleLoginBtn.addEventListener('click', handleGoogleAuth);

  const googleSignupBtn = document.getElementById('googleSignupBtn');
  if (googleSignupBtn) googleSignupBtn.addEventListener('click', handleGoogleAuth);

  // Handle redirect result after Google redirect sign-in
  try {
    const result = await getRedirectResult(auth);
    if (result && result.user) {
      window.location.href = '/';
    }
  } catch (redirectError) {
    // ignore if no redirect result present
    if (redirectError.code !== 'auth/no-auth-event') {
      console.error('Google redirect result error:', redirectError);
      showError("Google redirect result failed: " + redirectError.code + " — " + redirectError.message.replace('Firebase: ', ''));
    }
  }

  // Handle Logout
  const logoutBtn = document.getElementById('logoutBtn');
  if (logoutBtn) {
    logoutBtn.addEventListener('click', async () => {
      try {
        await signOut(auth);
        window.location.href = '/login';
      } catch (error) {
        console.error("Logout error", error);
      }
    });
  }

  // Global Auth State Observer (Route Guarding)
  onAuthStateChanged(auth, (user) => {
    const currentPath = window.location.pathname;
    const isAuthPage = currentPath === '/login' || currentPath === '/signup';

    if (user) {
      // User is signed in.
      if (isAuthPage) {
        // Don't let logged in users access login/signup pages
        window.location.href = '/';
      } else {
        // Unhide the main app content once we confirm they are logged in
        document.body.classList.remove('hidden-until-auth');
        // Update UI with user email if element exists
        const userEmailDisplay = document.getElementById('userEmailDisplay');
        if (userEmailDisplay) {
          userEmailDisplay.textContent = user.email;
        }
      }
    } else {
      // No user is signed in.
      if (!isAuthPage) {
        // Protect all non-auth routes → redirect to login
        window.location.href = '/login';
      } else {
        // Unhide the auth page content
        document.body.classList.remove('hidden-until-auth');
      }
    }
  });
});

