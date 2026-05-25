import React, { useState, useEffect, useMemo, useRef } from 'react';
import { RadialBarChart, RadialBar, PolarAngleAxis, ResponsiveContainer } from 'recharts';
import { Shield, Users, Activity, Zap, Rss, Clock, UserCheck } from 'lucide-react';
import io from 'socket.io-client';

// The address of your Python Socket.IO server
const SOCKET_ENDPOINT = 'http://localhost:5000';

// --- Main App Component ---
export default function App() {
  const [data, setData] = useState({
    chaosScore: 0,
    liveCount: 0,
    density: 0,
    flow: 0,
    surge: 0,
    latency: 0,
    status: 'Connecting...',
    log: [],
    activeVolunteerCount: 0,
  });

  const [error, setError] = useState(null);
  const [pollingInterval, setPollingInterval] = useState(5000); // This is now a UI setting, not for fetching
  const [isDispatchActive, setIsDispatchActive] = useState(false);
  
  // Use a ref to hold the socket instance to prevent re-connections on re-render
  const socketRef = useRef(null);
  // Use a ref to hold the score history to avoid re-rendering the component for every update
  const scoreHistoryRef = useRef([]);

  // --- Real-time Data Handling via WebSocket ---
  useEffect(() => {
    // Establish connection to the server
    socketRef.current = io(SOCKET_ENDPOINT);
    const socket = socketRef.current;

    socket.on('connect', () => {
      console.log('Dashboard connected to Vulcan server.');
      setError(null);
    });

    socket.on('disconnect', () => {
      console.log('Dashboard disconnected.');
      setError("Connection to Vulcan engine lost.");
      setData(prev => ({ ...prev, status: 'Disconnected', activeVolunteerCount: 0 }));
    });
    
    // Listen for the main data updates from the AI worker
    socket.on('update_data', (result) => {
      const newStatus = result.status.includes('HIGH') ? 'High Risk' : result.status.includes('MODERATE') ? 'Moderate Risk' : 'Safe';
      
      // Update score history
      scoreHistoryRef.current = [result.chaosScore, ...scoreHistoryRef.current].slice(0, 5); // Keep last 5 scores (10s window at 2s/update)
      const averageScore = scoreHistoryRef.current.reduce((a, b) => a + b, 0) / scoreHistoryRef.current.length;

      // --- Volunteer Dispatch Logic ---
      let dispatchSignal = isDispatchActive;
      if (averageScore >= 75 && !isDispatchActive) {
          setIsDispatchActive(true);
          dispatchSignal = true;
      } else if (averageScore < 40 && isDispatchActive) {
          setIsDispatchActive(false);
          dispatchSignal = false;
      }
      
      setData(prevData => {
        const newLogEntry = { time: new Date().toLocaleTimeString(), score: result.chaosScore.toFixed(1), status: newStatus };
        return {
          ...result,
          status: newStatus,
          log: (result.chaosScore.toFixed(1) !== prevData.chaosScore.toFixed(1) || newStatus !== prevData.status) 
               ? [newLogEntry, ...prevData.log].slice(0, 10) 
               : prevData.log,
          activeVolunteerCount: result.activeVolunteerCount,
        };
      });
    });

    // Listen for live updates on the volunteer count
    socket.on('volunteer_count_update', (countData) => {
        setData(prev => ({...prev, activeVolunteerCount: countData.count}));
    });

    // Cleanup on component unmount
    return () => {
      socket.disconnect();
    };
  }, [isDispatchActive]); // Re-run effect if dispatch status changes to send signal

  // This effect sends the dispatch signal when the status changes
  useEffect(() => {
      if (socketRef.current) {
          const dispatchPayload = {
              active: isDispatchActive,
              location: "Main Plaza",
              chaosScore: data.chaosScore,
              time: new Date().toLocaleTimeString()
          };
          socketRef.current.emit('dispatch_event', dispatchPayload);
      }
  }, [isDispatchActive, data.chaosScore]);


  const statusStyles = useMemo(() => {
    if (data.status === 'High Risk') return { bgColor: 'bg-red-900/50', borderColor: 'border-red-500', textColor: 'text-red-400', gaugeColor: '#ef4444' };
    if (data.status === 'Moderate Risk') return { bgColor: 'bg-yellow-900/50', borderColor: 'border-yellow-500', textColor: 'text-yellow-400', gaugeColor: '#f59e0b' };
    if (data.status === 'Disconnected' || data.status === 'Connecting...') return { bgColor: 'bg-slate-700/50', borderColor: 'border-slate-500', textColor: 'text-slate-400', gaugeColor: '#64748b' };
    return { bgColor: 'bg-green-900/50', borderColor: 'border-green-500', textColor: 'text-green-400', gaugeColor: '#22c55e' };
  }, [data.status]);

  return (
    <div className="bg-slate-900 text-slate-200 min-h-screen font-sans p-4 sm:p-6 lg:p-8">
      <div className="max-w-7xl mx-auto">
        <Header 
          status={data.status} 
          latency={data.latency} 
          statusColor={statusStyles.textColor}
        />
        
        {error && <div className="mt-6 p-4 rounded-xl border-2 border-red-500 bg-red-900/50 text-center text-red-300">{error}</div>}
        
        <div className={`mt-6 p-4 rounded-xl border-2 ${statusStyles.borderColor} ${statusStyles.bgColor} transition-all duration-500`}>
          <StatusAlert status={data.status} isDispatchActive={isDispatchActive} statusColor={statusStyles.textColor} />
        </div>

        <main className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-6">
          <div className="lg:col-span-2 bg-slate-800/50 p-6 rounded-xl border border-slate-700 flex flex-col items-center justify-center min-h-[300px]">
            <h2 className="text-xl font-bold text-slate-300 mb-4">CHAOS SCORE</h2>
            <MainScoreGauge score={data.chaosScore} color={statusStyles.gaugeColor} />
          </div>

          <div className="bg-slate-800/50 p-6 rounded-xl border border-slate-700">
            <LogPanel log={data.log} />
          </div>
        </main>

        <section className="grid grid-cols-2 md:grid-cols-4 gap-6 mt-6">
          <MetricCard icon={<Users size={24} className="text-sky-400" />} title="Live Count" value={data.liveCount.toFixed(0)} />
          <MetricCard icon={<UserCheck size={24} className="text-teal-400" />} title="Active Volunteers" value={data.activeVolunteerCount.toFixed(0)} />
          <MetricCard icon={<Zap size={24} className="text-amber-400" />} title="Flow Score" value={data.flow.toFixed(1)} unit="/ 100" />
          <MetricCard icon={<Shield size={24} className="text-indigo-400" />} title="Surge Score" value={data.surge.toFixed(1)} unit="/ 100" />
        </section>
      </div>
    </div>
  );
}

// --- Sub-Components ---
const Header = ({ status, latency, statusColor }) => ( <header className="flex flex-wrap justify-between items-center pb-4 border-b border-slate-700"><h1 className="text-3xl font-bold text-white tracking-wider"><Rss size={32} className="inline-block mr-3 text-sky-400" />VULCAN <span className="font-light text-slate-400">Live</span></h1><div className="flex items-center space-x-6 mt-4 sm:mt-0"><span className={`font-bold text-lg transition-colors duration-500 ${statusColor}`}>{status.toUpperCase()}</span><span className="text-sm text-slate-400 font-mono">Latency: {(latency || 0).toFixed(2)}s</span></div></header> );
const StatusAlert = ({ status, isDispatchActive, statusColor }) => { const messages = { 'Safe': 'Crowd conditions are normal.', 'Moderate Risk': 'Increased crowd density or chaotic flow detected.', 'High Risk': `Sustained high-risk event detected! Dispatch signal is ACTIVE.`, 'Connecting...': 'Attempting to connect to the Vulcan AI engine...', 'Disconnected': 'Connection to the AI engine has been lost.' }; return ( <div className="text-center"><h3 className={`text-2xl font-bold mb-1 transition-colors duration-500 ${statusColor}`}>{status.toUpperCase()}</h3><p className="text-slate-300">{isDispatchActive && status !== 'High Risk' ? 'Monitoring situation. Dispatch signal remains active.' : messages[status]}</p></div> ); };
const MainScoreGauge = ({ score, color }) => ( <ResponsiveContainer width="100%" height={300}><RadialBarChart innerRadius="70%" outerRadius="100%" data={[{ value: score }]} startAngle={180} endAngle={-180}><PolarAngleAxis type="number" domain={[0, 100]} angleAxisId={0} tick={false} /><RadialBar background dataKey="value" cornerRadius={10} fill={color} className="transition-all duration-500" /><text x="50%" y="50%" textAnchor="middle" dominantBaseline="middle" className="fill-current text-white font-bold text-6xl">{score.toFixed(1)}</text></RadialBarChart></ResponsiveContainer> );
const MetricCard = ({ icon, title, value, unit }) => ( <div className="bg-slate-800/50 p-4 rounded-xl border border-slate-700"><div className="flex items-center text-slate-400 mb-2">{icon}<h4 className="ml-2 font-semibold">{title}</h4></div><div className="text-4xl font-bold text-white">{value}{unit && <span className="text-xl font-normal text-slate-400 ml-1">{unit}</span>}</div></div> );
const LogPanel = ({ log }) => ( <div className="h-full flex flex-col"><h3 className="text-lg font-bold mb-4 text-slate-300">Event Log</h3><div className="flex-grow space-y-2 overflow-y-auto pr-2 max-h-[300px]">{log.length === 0 && <p className="text-slate-500">Awaiting system data...</p>}{log.map((entry, index) => { const logColor = entry.status.includes('High') ? 'text-red-400' : entry.status.includes('Moderate') ? 'text-yellow-400' : 'text-green-400'; return ( <div key={index} className="flex justify-between items-center text-sm font-mono bg-slate-900/50 p-2 rounded-md"><span className="text-slate-400">{entry.time}</span><span className={logColor}>Score: {entry.score}</span><span className={`font-bold ${logColor}`}>{entry.status.replace(/â🔴|🟡|🟢\s/g, '')}</span></div> ) })}</div></div> );

