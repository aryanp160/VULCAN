import React, { useState, useEffect, useRef } from 'react';
import { Siren, ShieldCheck, WifiOff, History, LogOut } from 'lucide-react';
import io from 'socket.io-client';

// The address of your Python Socket.IO server
const SOCKET_ENDPOINT = 'http://localhost:5000';

// --- Main Volunteer App Component ---
export default function App() {
  const [isRegistered, setIsRegistered] = useState(false);
  const [volunteerName, setVolunteerName] = useState('');
  const [alert, setAlert] = useState(null);
  const [alertHistory, setAlertHistory] = useState([]);
  const [error, setError] = useState(null);
  const [showHistory, setShowHistory] = useState(false);
  
  const socketRef = useRef(null);

  // This effect manages the WebSocket connection
  useEffect(() => {
    // Don't connect until the user has registered their name
    if (!isRegistered) return;

    // Establish connection to the server
    socketRef.current = io(SOCKET_ENDPOINT, {
      reconnectionAttempts: 5,
      reconnectionDelay: 2000,
    });
    const socket = socketRef.current;

    socket.on('connect', () => {
      console.log(`Volunteer '${volunteerName}' connected to Vulcan server.`);
      setError(null);
      // Once connected, register with the server and send the name
      socket.emit('register_volunteer', { name: volunteerName });
    });

    socket.on('disconnect', () => {
      console.log('Volunteer disconnected.');
      setError('Connection to server lost. Attempting to reconnect...');
    });

    // Listen for new dispatch alerts from the server
    socket.on('dispatch_alert', (alertData) => {
      if (alertData && alertData.active) {
        setAlert(alertData);
        // Add to history only if it's a new alert (check by time)
        setAlertHistory(prevHistory => {
            if (!prevHistory.some(h => h.time === alertData.time)) {
                return [alertData, ...prevHistory].slice(0, 20); // Keep last 20 alerts
            }
            return prevHistory;
        });
      } else {
        setAlert(null); // Alert has been cleared by the command center
      }
    });

    // Cleanup on component unmount or if user logs out
    return () => {
      socket.disconnect();
    };
  }, [isRegistered, volunteerName]);

  const handleRegister = (name) => {
    if (name.trim()) {
      setVolunteerName(name.trim());
      setIsRegistered(true);
    }
  };

  const handleLogout = () => {
    setIsRegistered(false);
    setVolunteerName('');
    setAlert(null);
    setAlertHistory([]);
    if(socketRef.current) socketRef.current.disconnect();
  }

  // --- Render logic ---
  if (!isRegistered) {
    return <LoginScreen onRegister={handleRegister} />;
  }

  // Show history overlay if toggled
  if (showHistory) {
    return <HistoryScreen history={alertHistory} name={volunteerName} onBack={() => setShowHistory(false)} onLogout={handleLogout}/>;
  }
  
  // Conditionally render main screens
  let mainScreen;
  if (error && !alert) {
    mainScreen = <ErrorScreen message={error} />;
  } else if (!alert) {
    mainScreen = <AllClearScreen />;
  } else {
    mainScreen = <AlertScreen alert={alert} />;
  }
  
  return (
    <div className="relative min-h-screen">
      {mainScreen}
      {/* Persistent top-right menu for history and logout */}
      <div className="absolute top-4 right-4 flex space-x-2">
        <button onClick={() => setShowHistory(true)} className="bg-slate-700/50 p-2 rounded-full text-white hover:bg-slate-600 transition-colors">
          <History size={24} />
        </button>
        <button onClick={handleLogout} className="bg-red-800/60 p-2 rounded-full text-white hover:bg-red-700 transition-colors">
          <LogOut size={24} />
        </button>
      </div>
    </div>
  );
}


// --- Screen Components ---

const LoginScreen = ({ onRegister }) => {
  const [name, setName] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    onRegister(name);
  };

  return (
    <div className="bg-slate-900 text-slate-300 min-h-screen flex flex-col items-center justify-center p-4">
      <Siren size={60} className="text-sky-400 mb-4" />
      <h1 className="text-4xl font-bold text-white mb-2">Vulcan Volunteer</h1>
      <p className="text-lg text-slate-400 mb-8">Please enter your name to register for duty.</p>
      <form onSubmit={handleSubmit} className="w-full max-w-sm">
        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Enter your name..."
          className="w-full bg-slate-800 border-2 border-slate-600 text-white text-xl text-center p-4 rounded-lg focus:ring-sky-500 focus:border-sky-500"
          required
        />
        <button
          type="submit"
          className="mt-4 w-full bg-sky-600 text-white font-bold text-xl py-4 rounded-lg hover:bg-sky-500 transition-colors"
        >
          Register for Duty
        </button>
      </form>
    </div>
  );
};

const HistoryScreen = ({ history, name, onBack, onLogout }) => (
  <div className="bg-slate-900 text-slate-300 min-h-screen p-4 flex flex-col">
    <div className="flex justify-between items-center mb-6">
       <h1 className="text-3xl font-bold text-white">Alert History</h1>
       <div className="flex items-center space-x-2">
        <span className="text-slate-400 hidden sm:inline">{name}</span>
        <button onClick={onLogout} className="bg-red-800/60 p-2 rounded-full text-white hover:bg-red-700 transition-colors">
          <LogOut size={24} />
        </button>
       </div>
    </div>
    <div className="flex-grow bg-slate-800/50 p-4 rounded-xl border border-slate-700 overflow-y-auto">
        {history.length === 0 ? (
            <p className="text-slate-500 text-center mt-8">No alerts received in this session.</p>
        ) : (
            <div className="space-y-3">
                {history.map((item, index) => (
                    <div key={index} className="bg-slate-900/70 p-3 rounded-lg border-l-4 border-red-500">
                        <div className="flex justify-between items-center font-mono text-sm">
                            <span className="font-bold text-red-400">{item.location}</span>
                            <span className="text-slate-400">{item.time}</span>
                        </div>
                        <p className="font-semibold text-white mt-1">Chaos Score: {Number(item.chaosScore).toFixed(1)}</p>
                    </div>
                ))}
            </div>
        )}
    </div>
     <button onClick={onBack} className="mt-6 w-full bg-sky-600 text-white font-bold text-xl py-4 rounded-lg hover:bg-sky-500 transition-colors">
        Back to Live Status
    </button>
  </div>
);

const ErrorScreen = ({ message }) => (
  <div className="bg-slate-800 text-slate-300 min-h-screen flex flex-col items-center justify-center p-4 text-center">
    <WifiOff size={80} className="text-yellow-500 mb-6" />
    <h1 className="text-4xl font-bold text-white mb-2">CONNECTION ERROR</h1>
    <p className="text-lg text-slate-400">{message}</p>
  </div>
);

const AllClearScreen = () => (
  <div className="bg-slate-800 text-slate-300 min-h-screen flex flex-col items-center justify-center p-4 text-center">
    <ShieldCheck size={80} className="text-green-500 mb-6" />
    <h1 className="text-4xl font-bold text-white mb-2">ALL CLEAR</h1>
    <p className="text-lg text-slate-400">Awaiting dispatch orders. Stay safe.</p>
  </div>
);

const AlertScreen = ({ alert }) => (
  <div className="bg-red-900 text-white min-h-screen flex flex-col items-center justify-center p-4 text-center animate-pulse-bg">
    <style>{`
      @keyframes pulse-bg { 0%, 100% { background-color: #450a0a; } 50% { background-color: #7f1d1d; } }
      .animate-pulse-bg { animation: pulse-bg 3s infinite; }
    `}</style>
    <div className="relative">
      <Siren size={80} className="text-white mb-6" />
      <div className="absolute top-0 left-0 w-full h-full rounded-full bg-white/50 animate-ping"></div>
    </div>
    <h1 className="text-5xl font-extrabold mb-4">DISPATCH ALERT</h1>
    <div className="bg-red-800/50 border-2 border-red-500 rounded-xl p-6 w-full max-w-md">
      <p className="text-xl text-red-200 mb-2">High-Risk Event Detected At:</p>
      <h2 className="text-4xl font-bold mb-4">{alert.location ? alert.location.toUpperCase() : 'N/A'}</h2>
      <p className="text-lg text-red-200">Current Chaos Score:</p>
      <p className="text-6xl font-bold mb-6">{Number(alert.chaosScore).toFixed(1)}</p>
      <p className="text-sm text-red-300">Dispatched at {alert.time}</p>
    </div>
  </div>
);

