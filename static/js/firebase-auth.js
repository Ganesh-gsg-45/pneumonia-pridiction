import { initializeApp } from "https://www.gstatic.com/firebasejs/10.8.0/firebase-app.js";
import { 
  getAuth, 
  createUserWithEmailAndPassword, 
  signInWithEmailAndPassword, 
  signInWithPopup,
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

document.addEventListener('DOMContentLoaded', () => {
  // Helper to show errors
  const showError = (message) => {
    const errorDiv = document.getElementById('authError');
    if (errorDiv) {
      errorDiv.textContent = message;
      errorDiv.classList.remove('hidden');
    }
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
      const err = document.getElementById('authError');

      btn.disabled = true;
      txt.classList.add('hidden');
      spin.classList.remove('hidden');
      err.classList.add('hidden');

      try {
        await signInWithEmailAndPassword(auth, email, password);
        window.location.href = '/'; // Redirect to main app
      } catch (error) {
        showError("Login failed: " + error.message.replace('Firebase: ', ''));
        btn.disabled = false;
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
      const err = document.getElementById('authError');

      if (password !== confirmPassword) {
        showError("Passwords do not match");
        return;
      }

      btn.disabled = true;
      txt.classList.add('hidden');
      spin.classList.remove('hidden');
      err.classList.add('hidden');

      try {
        await createUserWithEmailAndPassword(auth, email, password);
        window.location.href = '/'; // Redirect to main app
      } catch (error) {
        showError("Signup failed: " + error.message.replace('Firebase: ', ''));
        btn.disabled = false;
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
      showError("Google sign-in failed: " + error.message.replace('Firebase: ', ''));
    }
  };

  const googleLoginBtn = document.getElementById('googleLoginBtn');
  if (googleLoginBtn) googleLoginBtn.addEventListener('click', handleGoogleAuth);

  const googleSignupBtn = document.getElementById('googleSignupBtn');
  if (googleSignupBtn) googleSignupBtn.addEventListener('click', handleGoogleAuth);

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

