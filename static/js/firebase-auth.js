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
  const sendTokenToServer = async (idToken) => {
    try {
      const res = await fetch('/verify-token', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token: idToken }),
      });
      return res.ok;
    } catch (error) {
      console.error('Token verification error', error);
      return false;
    }
  };

  const verifyCurrentUser = async (user) => {
    if (!user) return false;
    try {
      const token = await user.getIdToken();
      return await sendTokenToServer(token);
    } catch (error) {
      console.error('Failed to verify current user token', error);
      return false;
    }
  };

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
        const user = auth.currentUser;
        const verified = await verifyCurrentUser(user);
        if (verified) {
          window.location.href = '/';
        } else {
          showError('Login failed: unable to verify authentication.');
          await auth.signOut();
          setButtonState(btn, false);
          txt.classList.remove('hidden');
          spin.classList.add('hidden');
        }
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
        const user = auth.currentUser;
        const verified = await verifyCurrentUser(user);
        if (verified) {
          window.location.href = '/';
        } else {
          showError('Signup failed: unable to verify authentication.');
          await auth.signOut();
          setButtonState(btn, false);
          txt.classList.remove('hidden');
          spin.classList.add('hidden');
        }
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
      const user = auth.currentUser;
      const verified = await verifyCurrentUser(user);
      if (verified) {
        window.location.href = '/';
      } else {
        showError('Google sign-in failed: unable to verify authentication.');
        await auth.signOut();
      }
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
      const verified = await verifyCurrentUser(result.user);
      if (verified) {
        window.location.href = '/';
      } else {
        showError('Google sign-in failed: unable to verify authentication.');
        await auth.signOut();
      }
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
        window.location.href = '/logout';
      } catch (error) {
        console.error("Logout error", error);
        window.location.href = '/logout';
      }
    });
  }

  // Global Auth State Observer (Route Guarding)
  onAuthStateChanged(auth, async (user) => {
    const currentPath = window.location.pathname;
    const isAuthPage = currentPath === '/login' || currentPath === '/signup';

    if (user) {
      const verified = await verifyCurrentUser(user);
      if (!verified) {
        await auth.signOut();
        if (!isAuthPage) {
          window.location.href = '/login';
        } else {
          document.body.classList.remove('hidden-until-auth');
        }
        return;
      }

      if (isAuthPage) {
        window.location.href = '/';
        return;
      }

      document.body.classList.remove('hidden-until-auth');
      const userEmailDisplay = document.getElementById('userEmailDisplay');
      if (userEmailDisplay) {
        userEmailDisplay.textContent = user.email;
      }
    } else {
      if (!isAuthPage) {
        window.location.href = '/login';
      } else {
        document.body.classList.remove('hidden-until-auth');
      }
    }
  });
});

