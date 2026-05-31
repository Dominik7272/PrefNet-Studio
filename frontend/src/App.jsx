import React, { useState, useEffect, useRef } from 'react';
import { 
  Play, Square, RefreshCw, Folder, Trash, Save, Plus, ArrowLeft, 
  ArrowRight, SkipForward, RotateCcw, AlertTriangle, Layers, 
  Activity, Grid, Settings, Database, Sliders, ChevronRight, 
  FileText, Download, X, Home, HardDrive, Cpu, Check, CheckSquare, 
  Search, Eye, HelpCircle, ArrowUpRight, ArrowUpDown, ChevronDown, CheckCircle, GripVertical,
  Hash, Percent, Zap
} from 'lucide-react';

const API_BASE = window.location.origin.includes('5173') ? 'http://localhost:8000' : '';

const DEFAULT_NN_PRESETS = [
  {
    name: "BiLabel Standard (Recommended)",
    layers: [
      { type: 'Linear', out_features: 512 },
      { type: 'BatchNorm1d', num_features: 512 },
      { type: 'ReLU' },
      { type: 'Dropout', p: 0.2 },
      { type: 'Linear', out_features: 1 },
      { type: 'Sigmoid' }
    ]
  },
  {
    name: "Deep & Narrow",
    layers: [
      { type: 'Linear', out_features: 256 },
      { type: 'ReLU' },
      { type: 'Linear', out_features: 128 },
      { type: 'ReLU' },
      { type: 'Linear', out_features: 64 },
      { type: 'ReLU' },
      { type: 'Linear', out_features: 1 },
      { type: 'Sigmoid' }
    ]
  },
  {
    name: "Wide & Shallow",
    layers: [
      { type: 'Linear', out_features: 1024 },
      { type: 'ReLU' },
      { type: 'Dropout', p: 0.3 },
      { type: 'Linear', out_features: 1 },
      { type: 'Sigmoid' }
    ]
  },
  {
    name: "Leaky Residual-like",
    layers: [
      { type: 'Linear', out_features: 512 },
      { type: 'BatchNorm1d', num_features: 512 },
      { type: 'LeakyReLU' },
      { type: 'Dropout', p: 0.2 },
      { type: 'Linear', out_features: 256 },
      { type: 'BatchNorm1d', num_features: 256 },
      { type: 'LeakyReLU' },
      { type: 'Linear', out_features: 1 },
      { type: 'Sigmoid' }
    ]
  }
];

export default function App() {
  // Navigation & Workspace State
  const [workspace, setWorkspace] = useState(null);
  const [recents, setRecents] = useState([]);
  const [activeTab, setActiveTab] = useState('dashboard'); // 'dashboard' | 'dataset' | 'labeler' | 'trainer' | 'predictions' | 'pipelines'

  const activeDs = workspace && workspace.settings && workspace.settings.datasets
    ? (workspace.settings.datasets.find(d => d.name === workspace.settings.active_dataset) || workspace.settings.datasets[0] || { name: 'default', label_a: 'Liked', label_b: 'Disliked' })
    : { name: 'default', label_a: 'Liked', label_b: 'Disliked' };
  const [loading, setLoading] = useState(true);
  const [errorMsg, setErrorMsg] = useState(null);

  // File Browser Modal State
  const [showBrowser, setShowBrowser] = useState(false);
  const [browserPath, setBrowserPath] = useState('');
  const [browserData, setBrowserData] = useState({ folders: [], current_path: '', parent_path: '', drives: [] });
  const [browserCallback, setBrowserCallback] = useState(null);
  const [browserMultiSelect, setBrowserMultiSelect] = useState(false);
  const [selectedFolders, setSelectedFolders] = useState([]);
  const [lastBrowsedPath, setLastBrowsedPath] = useState(() => {
    return localStorage.getItem('last_browsed_path') || '';
  });
  const [customAlert, setCustomAlert] = useState({ show: false, message: '', title: 'Error' });

  const [useCustomNN, setUseCustomNN] = useState(false);
  const [customLayers, setCustomLayers] = useState([]);
  const [presets, setPresets] = useState([]);
  const [newPresetName, setNewPresetName] = useState('');
  const [predictViewMode, setPredictViewMode] = useState('dynamic'); // 'square' | 'dynamic' | 'list'
  const [updateAvailable, setUpdateAvailable] = useState(false);
  const [latestVersion, setLatestVersion] = useState('');
  const [dragOverNodeId, setDragOverNodeId] = useState(null);
  const [selectedLayerIdx, setSelectedLayerIdx] = useState(null);
  const [dragOverLayerIdx, setDragOverLayerIdx] = useState(null);
  const [dragOverInsertIdx, setDragOverInsertIdx] = useState(null);

  const showCustomAlert = (message, title = 'Error') => {
    setCustomAlert({ show: true, message, title });
  };

  const getParentOfPath = (path) => {
    if (!path) return '';
    const cleanPath = path.replace(/\\/g, '/');
    if (/^[A-Za-z]:\/?$/.test(cleanPath)) {
      return path;
    }
    const idx = Math.max(cleanPath.lastIndexOf('/'), cleanPath.lastIndexOf('\\'));
    if (idx <= 0) {
      return cleanPath.startsWith('/') ? '/' : '';
    }
    const parent = path.substring(0, idx);
    if (/^[A-Za-z]:$/.test(parent)) {
      return parent + (path.includes('\\') ? '\\' : '/');
    }
    return parent;
  };

  const getActiveDatasetData = () => {
    return activeDs;
  };

  // Welcome Modals
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newWorkspaceName, setNewWorkspaceName] = useState('');
  const [newWorkspacePath, setNewWorkspacePath] = useState('');
  const [showOpenModal, setShowOpenModal] = useState(false);
  const [openWorkspacePath, setOpenWorkspacePath] = useState('');

  // Labeled gallery view inside Dataset Studio (Auditing feature)
  const [auditClass, setAuditClass] = useState('a'); // 'a' or 'b'
  const [auditImages, setAuditImages] = useState([]);
  const [loadingAudit, setLoadingAudit] = useState(false);

  // Interactive Labeler State
  const [labelFolder, setLabelFolder] = useState('');
  const [labelQueue, setLabelQueue] = useState([]);
  const [labelIndex, setLabelIndex] = useState(0);
  const [labelHistory, setLabelHistory] = useState([]);
  const [labelRecursive, setLabelRecursive] = useState(true);
  
  // Custom Zoom & Pan State for Labeler Viewport
  const [zoomScale, setZoomScale] = useState(1);
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  const labelViewportRef = useRef(null);

  // Optimizer & Training State
  const [trainModelName, setTrainModelName] = useState('model_v1');
  const [trainEpochs, setTrainEpochs] = useState(20);
  const [trainBatchSize, setTrainBatchSize] = useState(32);
  const [trainLearningRate, setTrainLearningRate] = useState(0.001);
  const [isTraining, setIsTraining] = useState(false);
  const [trainProgress, setTrainProgress] = useState(0);
  const [trainLog, setTrainLog] = useState([]);
  const [trainStats, setTrainStats] = useState([]);
  const logEndRef = useRef(null);

  // Predictions State
  const [predictFolder, setPredictFolder] = useState('');
  const [predictModel, setPredictModel] = useState('');
  const [predictQueue, setPredictQueue] = useState([]);
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictProgress, setPredictProgress] = useState(0);
  const [predictLog, setPredictLog] = useState([]);
  const [predictFilter, setPredictFilter] = useState('all');
  const [predictSort, setPredictSort] = useState('score_desc');
  const [selectedPredictItem, setSelectedPredictItem] = useState(null);
  const [selectedPredictPaths, setSelectedPredictPaths] = useState([]); // for bulk actions

  // Automated pipelines batch script state
  const [batchJobs, setBatchJobs] = useState([]);
  const [isBatchRunning, setIsBatchRunning] = useState(false);
  const [batchProgress, setBatchProgress] = useState(0);
  const [batchStatusText, setBatchStatusText] = useState('Ready');
  const [batchLog, setBatchLog] = useState([]);

  // Fetch workspaces & config
  useEffect(() => {
    fetchRecents();
    checkActiveWorkspace();
    checkUpdates();
  }, []);

  useEffect(() => {
    if (workspace && workspace.settings) {
      const customL = workspace.settings.custom_layers;
      
      if (Array.isArray(customL)) {
        setUseCustomNN(true);
        const filteredL = customL.filter(l => l && typeof l.type === 'string');
        setCustomLayers(filteredL);
      } else {
        setUseCustomNN(false);
        const defaultLayers = [
          { type: 'Linear', out_features: 512 },
          { type: 'BatchNorm1d', num_features: 512 },
          { type: 'ReLU' },
          { type: 'Dropout', p: 0.2 },
          { type: 'Linear', out_features: 1 },
          { type: 'Sigmoid' }
        ];
        setCustomLayers(defaultLayers);
      }
    }
  }, [workspace ? workspace.path : null]);

  useEffect(() => {
    let parsed = [];
    try {
      const saved = localStorage.getItem('nn_presets');
      if (saved) {
        const val = JSON.parse(saved);
        if (Array.isArray(val)) {
          parsed = val.filter(item => item && typeof item === 'object' && typeof item.name === 'string');
        }
      }
    } catch (e) {
      console.error("Failed to parse nn_presets from localStorage", e);
    }
    setPresets([...DEFAULT_NN_PRESETS, ...parsed]);
  }, []);

  // Handle keyboard hotkeys for labeler tab
  useEffect(() => {
    if (activeTab !== 'labeler') return;
    
    const handleKeyDown = (e) => {
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement.tagName)) {
        return;
      }
      
      if (e.key === 'ArrowRight') {
        e.preventDefault();
        executeLabelAction('a');
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        executeLabelAction('b');
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        executeLabelAction('skip');
      } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        if (labelHistory.length > 0) {
          undoLabelAction();
        }
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [activeTab, labelQueue, labelIndex, labelHistory, workspace]);

  // Non-passive wheel event listener for Zoom without page scrolling
  useEffect(() => {
    const el = labelViewportRef.current;
    if (!el) return;

    const onWheel = (e) => {
      e.preventDefault();
      if (e.deltaY < 0) {
        setZoomScale(prev => Math.min(10, prev * 1.15));
      } else {
        setZoomScale(prev => Math.max(0.1, prev / 1.15));
      }
    };

    el.addEventListener('wheel', onWheel, { passive: false });
    return () => {
      el.removeEventListener('wheel', onWheel);
    };
  }, [activeTab, labelQueue, labelIndex]);

  const fetchRecents = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/workspaces/recents`);
      const data = await res.json();
      setRecents(data || []);
    } catch (e) {
      console.error("Error fetching recents", e);
    }
  };

  const checkUpdates = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/system/update-check`);
      const data = await res.json();
      if (data && data.update_available) {
        setUpdateAvailable(true);
        setLatestVersion(data.latest_version);
      }
    } catch (e) {
      console.error("Error checking updates", e);
    }
  };

  const checkActiveWorkspace = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/workspace/info`);
      const data = await res.json();
      if (data.active) {
        setWorkspace(data);
        if (data.settings) {
          setTrainEpochs(data.settings.epochs || 20);
          setTrainBatchSize(data.settings.batch_size || 32);
          setTrainLearningRate(data.settings.learning_rate || 0.001);
          setTrainModelName(data.settings.load_path ? data.settings.load_path.replace('.pth', '') : 'model_v1');
          setPredictModel(data.settings.load_path ? data.settings.load_path.replace('.pth', '') : '');
        }
        if (data.scripts) {
          setBatchJobs(data.scripts);
        }
      }
    } catch (e) {
      console.error("Error loading workspace info", e);
    } finally {
      setLoading(false);
    }
  };

  // Triggers Directory Browser modal
  const openDirectoryBrowser = (initialPath, callback, allowMulti = false) => {
    setBrowserCallback(() => callback);
    setBrowserMultiSelect(allowMulti);
    setSelectedFolders([]);
    let targetPath = '';
    if (initialPath) {
      const firstPath = initialPath.split(';')[0];
      const parent = getParentOfPath(firstPath);
      targetPath = parent || firstPath;
    } else {
      targetPath = lastBrowsedPath || '';
    }
    fetchBrowserData(targetPath);
    setShowBrowser(true);
  };

  const fetchBrowserData = async (path) => {
    try {
      const res = await fetch(`${API_BASE}/api/fs/browse?path=${encodeURIComponent(path)}`);
      const data = await res.json();
      if (data.error) {
        showCustomAlert(data.error, "Access Denied");
      } else {
        setBrowserData(data);
        setBrowserPath(data.current_path);
        setSelectedFolders([]);
      }
    } catch (e) {
      console.error("Error browsing files", e);
    }
  };

  const selectBrowsedFolder = () => {
    if (browserCallback) {
      if (browserMultiSelect && selectedFolders.length > 0) {
        browserCallback(selectedFolders.join(';'));
        const firstPath = selectedFolders[0];
        const parent = getParentOfPath(firstPath);
        if (parent) {
          setLastBrowsedPath(parent);
          localStorage.setItem('last_browsed_path', parent);
        }
      } else {
        browserCallback(browserPath);
        const parent = getParentOfPath(browserPath);
        if (parent) {
          setLastBrowsedPath(parent);
          localStorage.setItem('last_browsed_path', parent);
        }
      }
    }
    setShowBrowser(false);
  };

  // Workspace Actions
  const handleOpenWorkspace = async (path) => {
    try {
      setErrorMsg(null);
      const res = await fetch(`${API_BASE}/api/workspaces/open`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path })
      });
      const data = await res.json();
      if (data.error) {
        setErrorMsg(data.error);
      } else {
        setWorkspace(data);
        setShowOpenModal(false);
        setActiveTab('dashboard');
        checkActiveWorkspace();
      }
    } catch (e) {
      setErrorMsg(e.toString());
    }
  };

  const handleCreateWorkspace = async () => {
    if (!newWorkspaceName || !newWorkspacePath) {
      showCustomAlert("Please fill in all fields.");
      return;
    }
    try {
      setErrorMsg(null);
      const res = await fetch(`${API_BASE}/api/workspaces/create`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: newWorkspaceName, path: newWorkspacePath })
      });
      const data = await res.json();
      if (data.error) {
        setErrorMsg(data.error);
      } else {
        setWorkspace(data);
        setShowCreateModal(false);
        setActiveTab('dashboard');
        checkActiveWorkspace();
      }
    } catch (e) {
      setErrorMsg(e.toString());
    }
  };

  const saveWorkspaceSettings = async (updatedSettings, updatedScripts = null) => {
    if (!workspace) return;
    try {
      const res = await fetch(`${API_BASE}/api/workspace/save-settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          settings: updatedSettings,
          scripts: updatedScripts !== null ? updatedScripts : batchJobs
        })
      });
      const data = await res.json();
      if (data.success) {
        setWorkspace(prev => ({
          ...prev,
          settings: data.settings,
          stats: data.stats
        }));
      }
    } catch (e) {
      console.error("Error saving configuration changes", e);
    }
  };

  // Dataset Studio Operations
  const [dsName, setDsName] = useState('');
  const [dsLabelA, setDsLabelA] = useState('Liked');
  const [dsLabelB, setDsLabelB] = useState('Disliked');

  const addDatasetProfile = async () => {
    if (!dsName) return;
    try {
      const res = await fetch(`${API_BASE}/api/workspace/dataset/add`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: dsName, label_a: dsLabelA, label_b: dsLabelB })
      });
      const data = await res.json();
      if (data.error) {
        showCustomAlert(data.error);
      } else {
        setWorkspace(prev => ({ ...prev, settings: data.settings, stats: data.stats }));
        setDsName('');
      }
    } catch (e) {
      console.error(e);
    }
  };

  const deleteDatasetProfile = async (name) => {
    const confirmDelete = window.confirm(`Are you sure you want to delete dataset profile "${name}"? associate folders will be removed from your disk.`);
    if (!confirmDelete) return;
    try {
      const res = await fetch(`${API_BASE}/api/workspace/dataset/delete`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, delete_files: true })
      });
      const data = await res.json();
      if (data.error) {
        showCustomAlert(data.error);
      } else {
        setWorkspace(prev => ({ ...prev, settings: data.settings, stats: data.stats }));
      }
    } catch (e) {
      console.error(e);
    }
  };

  const activateDatasetProfile = async (name) => {
    try {
      const res = await fetch(`${API_BASE}/api/workspace/dataset/activate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name })
      });
      const data = await res.json();
      if (data.error) {
        showCustomAlert(data.error);
      } else {
        setWorkspace(prev => ({ ...prev, settings: data.settings, stats: data.stats }));
      }
    } catch (e) {
      console.error(e);
    }
  };

  const saveDatasetProfileChanges = async (oldName, name, labelA, labelB) => {
    try {
      const res = await fetch(`${API_BASE}/api/workspace/dataset/save`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ old_name: oldName, name, label_a: labelA, label_b: labelB })
      });
      const data = await res.json();
      if (data.error) {
        showCustomAlert(data.error);
      } else {
        setWorkspace(prev => ({ ...prev, settings: data.settings, stats: data.stats }));
      }
    } catch (e) {
      console.error(e);
    }
  };

  // Load visual collection gallery for auditing
  const fetchAuditGallery = async (labelClassKey) => {
    if (!workspace) return;
    setLoadingAudit(true);
    setAuditClass(labelClassKey);
    try {
      const res = await fetch(`${API_BASE}/api/workspace/dataset/gallery?dataset=${workspace.settings.active_dataset}&label_class=${labelClassKey}`);
      const data = await res.json();
      setAuditImages(data.images || []);
    } catch (e) {
      console.error("Error loading audit gallery", e);
    } finally {
      setLoadingAudit(false);
    }
  };

  useEffect(() => {
    if (activeTab === 'dataset' && workspace) {
      fetchAuditGallery(auditClass);
    }
  }, [activeTab, workspace ? workspace.settings.active_dataset : null]);

  const handleAuditAction = async (imgPath, action) => {
    // action: 'delete' or 'move'
    const confirmAction = action === 'delete' ? window.confirm("Delete this image permanently from dataset?") : true;
    if (!confirmAction) return;
    try {
      const res = await fetch(`${API_BASE}/api/workspace/dataset/gallery/action`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset: workspace.settings.active_dataset,
          label_class: auditClass,
          image_path: imgPath,
          action
        })
      });
      const data = await res.json();
      if (data.error) {
        showCustomAlert(data.error);
      } else {
        setWorkspace(prev => ({ ...prev, stats: data.stats }));
        setAuditImages(prev => prev.filter(p => p !== imgPath));
      }
    } catch (e) {
      console.error(e);
    }
  };

  // Interactive Labeler Setup
  const loadLabelQueue = async () => {
    if (!labelFolder) return;
    try {
      const res = await fetch(`${API_BASE}/api/workspace/label/queue?folder=${encodeURIComponent(labelFolder)}&recursive=${labelRecursive}`);
      const data = await res.json();
      if (data.error) {
        showCustomAlert(data.error);
      } else {
        setLabelQueue(data.paths || []);
        setLabelIndex(0);
        setLabelHistory([]);
        resetViewportZoom();
      }
    } catch (e) {
      console.error(e);
    }
  };

  const executeLabelAction = async (action) => {
    if (labelQueue.length === 0 || labelIndex >= labelQueue.length) return;
    const currentSrc = labelQueue[labelIndex];
    const mode = workspace.settings.copy_move_mode || 'copy';
    
    try {
      const res = await fetch(`${API_BASE}/api/workspace/label/action`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ src: currentSrc, action, mode })
      });
      const data = await res.json();
      if (data.error) {
        showCustomAlert(data.error);
      } else {
        setLabelHistory(prev => {
          const idx = prev.findIndex(h => h.index === labelIndex);
          if (idx !== -1) {
            const nextHistory = [...prev];
            nextHistory[idx] = { ...nextHistory[idx], action };
            return nextHistory;
          } else {
            return [...prev, { dest: action === 'skip' ? null : 'active', src: currentSrc, action, index: labelIndex }];
          }
        });
        setLabelIndex(prev => prev + 1);
        setWorkspace(prev => ({ ...prev, stats: data.stats }));
        resetViewportZoom();
      }
    } catch (e) {
      console.error(e);
    }
  };

  const undoLabelAction = async () => {
    if (labelHistory.length === 0) return;
    try {
      const res = await fetch(`${API_BASE}/api/workspace/label/undo`, {
        method: 'POST'
      });
      const data = await res.json();
      if (data.error) {
        showCustomAlert(data.error);
      } else {
        const lastAction = labelHistory[labelHistory.length - 1];
        setLabelIndex(lastAction.index);
        setLabelHistory(prev => prev.slice(0, -1));
        setWorkspace(prev => ({ ...prev, stats: data.stats }));
        resetViewportZoom();
      }
    } catch (e) {
      console.error(e);
    }
  };

  const undoToSpecificHistoryIndex = async (historyIndex) => {
    const stepsToUndo = labelHistory.length - historyIndex;
    if (stepsToUndo <= 0) return;
    
    // We execute standard undo sequentially in a loop to revert steps securely
    for (let i = 0; i < stepsToUndo; i++) {
      await undoLabelAction();
    }
  };

  // Zoom & Pan Mouse Helpers
  const resetViewportZoom = () => {
    setZoomScale(1);
    setPanOffset({ x: 0, y: 0 });
  };

  const handleZoomWheel = (e) => {
    e.preventDefault();
    if (e.deltaY < 0) {
      setZoomScale(prev => Math.min(10, prev * 1.15));
    } else {
      setZoomScale(prev => Math.max(0.1, prev / 1.15));
    }
  };

  const handlePanStart = (e) => {
    if (e.button !== 0) return; // Left click only
    setIsPanning(true);
    setPanStart({ x: e.clientX - panOffset.x, y: e.clientY - panOffset.y });
  };

  const handlePanning = (e) => {
    if (!isPanning) return;
    setPanOffset({
      x: e.clientX - panStart.x,
      y: e.clientY - panStart.y
    });
  };

  const handlePanEnd = () => {
    setIsPanning(false);
  };

  // NN Training Setup
  const startModelTraining = async () => {
    if (isTraining) return;
    setIsTraining(true);
    setTrainProgress(0);
    setTrainLog(['Requesting training start...']);
    setTrainStats([]);
    
    try {
      const res = await fetch(`${API_BASE}/api/workspace/train/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_name: trainModelName })
      });
      const data = await res.json();
      if (data.error) {
        setTrainLog(prev => [...prev, `[ERROR] ${data.error}`]);
        setIsTraining(false);
      }
    } catch (e) {
      setTrainLog(prev => [...prev, `[ERROR] ${e.toString()}`]);
      setIsTraining(false);
    }
  };

  const stopModelTraining = async () => {
    try {
      await fetch(`${API_BASE}/api/workspace/train/stop`, { method: 'POST' });
    } catch (e) {
      console.error(e);
    }
  };

  // Training Poller
  useEffect(() => {
    let timer;
    if (isTraining) {
      timer = setInterval(async () => {
        try {
          const res = await fetch(`${API_BASE}/api/workspace/train/status`);
          const data = await res.json();
          setTrainProgress(data.progress || 0);
          setTrainLog(data.log || []);
          setTrainStats(data.epoch_stats || []);
          
          if (data.status !== 'running') {
            setIsTraining(false);
            clearInterval(timer);
            checkActiveWorkspace();
          }
        } catch (e) {
          console.error(e);
        }
      }, 1000);
    }
    return () => clearInterval(timer);
  }, [isTraining]);

  const saveCustomLayersSettings = (layersList, enabled = useCustomNN, graph = null) => {
    if (!workspace || !workspace.settings) return;
    const nextSettings = {
      ...workspace.settings,
      custom_layers: enabled ? layersList : null,
      custom_layers_graph: null
    };
    saveWorkspaceSettings(nextSettings);
  };

  const toggleUseCustomNN = (enabled) => {
    setUseCustomNN(enabled);
    saveCustomLayersSettings(customLayers, enabled, null);
  };

  const updateLayerParam = (idx, key, value) => {
    setCustomLayers(prev => {
      const next = prev.map((layer, i) => {
        if (i === idx) {
          return { ...layer, [key]: value };
        }
        return layer;
      });
      saveCustomLayersSettings(next, useCustomNN, null);
      return next;
    });
  };

  const removeLayer = (idx) => {
    setCustomLayers(prev => {
      const next = prev.filter((_, i) => i !== idx);
      saveCustomLayersSettings(next, useCustomNN, null);
      return next;
    });
    setSelectedLayerIdx(null);
  };

  const insertLayer = (insertIdx, type) => {
    const defaultParams = {};
    if (type === 'Linear') {
      defaultParams.out_features = 512;
      defaultParams.bias = true;
    } else if (type === 'BatchNorm1d') {
      defaultParams.num_features = '';
      defaultParams.eps = 1e-5;
      defaultParams.momentum = 0.1;
    } else if (type === 'Dropout') {
      defaultParams.p = 0.2;
      defaultParams.inplace = false;
    } else if (type === 'ReLU' || type === 'LeakyReLU') {
      defaultParams.inplace = false;
      if (type === 'LeakyReLU') {
        defaultParams.negative_slope = 0.01;
      }
    }
    
    setCustomLayers(prev => {
      const next = [...prev];
      next.splice(insertIdx, 0, { type, ...defaultParams });
      saveCustomLayersSettings(next, useCustomNN, null);
      return next;
    });
    setSelectedLayerIdx(insertIdx);
  };

  const moveLayer = (srcIdx, destIdx) => {
    if (srcIdx === destIdx || srcIdx === destIdx - 1) return;
    setCustomLayers(prev => {
      const next = [...prev];
      const [moved] = next.splice(srcIdx, 1);
      
      let targetIdx = destIdx;
      if (srcIdx < destIdx) {
        targetIdx = destIdx - 1;
      }
      
      next.splice(targetIdx, 0, moved);
      saveCustomLayersSettings(next, useCustomNN, null);
      
      if (selectedLayerIdx === srcIdx) {
        setSelectedLayerIdx(targetIdx);
      } else if (selectedLayerIdx > srcIdx && selectedLayerIdx <= targetIdx) {
        setSelectedLayerIdx(prevSelected => prevSelected - 1);
      } else if (selectedLayerIdx < srcIdx && selectedLayerIdx >= targetIdx) {
        setSelectedLayerIdx(prevSelected => prevSelected + 1);
      }
      
      return next;
    });
  };

  const replaceLayer = (idx, type) => {
    const defaultParams = {};
    if (type === 'Linear') {
      defaultParams.out_features = 512;
      defaultParams.bias = true;
    } else if (type === 'BatchNorm1d') {
      defaultParams.num_features = '';
      defaultParams.eps = 1e-5;
      defaultParams.momentum = 0.1;
    } else if (type === 'Dropout') {
      defaultParams.p = 0.2;
      defaultParams.inplace = false;
    } else if (type === 'ReLU' || type === 'LeakyReLU') {
      defaultParams.inplace = false;
      if (type === 'LeakyReLU') {
        defaultParams.negative_slope = 0.01;
      }
    }
    
    setCustomLayers(prev => {
      const next = prev.map((layer, i) => {
        if (i === idx) {
          return { type, ...defaultParams };
        }
        return layer;
      });
      saveCustomLayersSettings(next, useCustomNN, null);
      return next;
    });
  };

  const handleSavePreset = () => {
    if (!newPresetName.trim()) return;
    const newPreset = {
      name: newPresetName.trim(),
      layers: customLayers,
      graph: { nodes: graphNodes, connections: graphConnections }
    };
    let parsed = [];
    try {
      const saved = localStorage.getItem('nn_presets');
      if (saved) {
        const val = JSON.parse(saved);
        if (Array.isArray(val)) {
          parsed = val.filter(item => item && typeof item === 'object' && typeof item.name === 'string');
        }
      }
    } catch (e) {
      console.error(e);
    }
    const nextSaved = [...parsed, newPreset];
    localStorage.setItem('nn_presets', JSON.stringify(nextSaved));
    setPresets([...DEFAULT_NN_PRESETS, ...nextSaved]);
    setNewPresetName('');
    showCustomAlert(`Preset "${newPreset.name}" saved globally!`, "Success");
  };



  // Predictions / Inference Setup
  const loadPredictQueue = async () => {
    if (!predictFolder) return;
    try {
      const res = await fetch(`${API_BASE}/api/workspace/label/queue?folder=${encodeURIComponent(predictFolder)}&recursive=true`);
      const data = await res.json();
      if (data.error) {
        showCustomAlert(data.error);
      } else {
        const queueItems = (data.paths || [])
          .filter(p => typeof p === 'string' && p.trim() !== '')
          .map(p => ({
            path: p,
            score: null,
            label: 'Pending',
            label_key: ''
          }));
        setPredictQueue(queueItems);
        setSelectedPredictPaths([]);
      }
    } catch (e) {
      console.error(e);
    }
  };

  const startInference = async () => {
    if (isPredicting || predictQueue.length === 0 || !predictModel) return;
    setIsPredicting(true);
    setPredictProgress(0);
    setPredictLog(['Starting batch inference thread...']);
    
    try {
      const res = await fetch(`${API_BASE}/api/workspace/predict/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          paths: predictQueue.filter(item => item && item.path).map(item => item.path),
          model: predictModel
        })
      });
      const data = await res.json();
      if (data.error) {
        showCustomAlert(data.error);
        setIsPredicting(false);
      }
    } catch (e) {
      console.error(e);
      setIsPredicting(false);
    }
  };

  const stopInference = async () => {
    try {
      await fetch(`${API_BASE}/api/workspace/predict/stop`, { method: 'POST' });
    } catch (e) {
      console.error(e);
    }
  };

  // Inference status poller
  useEffect(() => {
    let timer;
    if (isPredicting) {
      timer = setInterval(async () => {
        try {
          const res = await fetch(`${API_BASE}/api/workspace/predict/status`);
          const data = await res.json();
          setPredictProgress(data.progress || 0);
          setPredictLog(data.log || []);
          
          if (data.results && data.results.length > 0) {
            const thresh = (workspace.settings.threshold || 50) / 100.0;
            const labelA = activeDs.label_a;
            const labelB = activeDs.label_b;
            
            setPredictQueue(prev => {
              const nextQueue = [...prev];
              data.results.forEach(resItem => {
                const qIdx = nextQueue.findIndex(item => item.path === resItem.path);
                if (qIdx !== -1) {
                  if (resItem.score !== null) {
                    const lKey = resItem.score >= thresh ? 'a' : 'b';
                    nextQueue[qIdx] = {
                      path: resItem.path,
                      score: resItem.score,
                      label_key: lKey,
                      label: lKey === 'a' ? labelA : labelB
                    };
                  } else {
                    nextQueue[qIdx] = {
                      path: resItem.path,
                      score: null,
                      label_key: 'error',
                      label: 'Error'
                    };
                  }
                }
              });
              return nextQueue;
            });
          }
          
          if (data.status !== 'running') {
            setIsPredicting(false);
            clearInterval(timer);
          }
        } catch (e) {
          console.error(e);
        }
      }, 1000);
    }
    return () => clearInterval(timer);
  }, [isPredicting, workspace]);

  const handlePredictOverride = async (item, action) => {
    const copyMoveMode = workspace.settings.copy_move_mode || 'copy';
    try {
      const res = await fetch(`${API_BASE}/api/workspace/predict/action`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ src: item.path, action, mode: copyMoveMode })
      });
      const data = await res.json();
      if (data.error) {
        showCustomAlert(data.error);
      } else {
        setPredictQueue(prev => prev.map(p => {
          if (p.path === item.path) {
            return {
              ...p,
              label_key: action,
              label: action === 'a' ? activeDs.label_a : activeDs.label_b
            };
          }
          return p;
        }));
        setWorkspace(prev => ({ ...prev, stats: data.stats }));
        if (selectedPredictItem && selectedPredictItem.path === item.path) {
          setSelectedPredictItem(prev => ({
            ...prev,
            label_key: action,
            label: action === 'a' ? activeDs.label_a : activeDs.label_b
          }));
        }
      }
    } catch (e) {
      console.error(e);
    }
  };

  // Bulk overrides selection action
  const togglePredictItemSelection = (path) => {
    setSelectedPredictPaths(prev => {
      if (prev.includes(path)) {
        return prev.filter(p => p !== path);
      } else {
        return [...prev, path];
      }
    });
  };

  const handleBulkPredictionAction = async (action) => {
    if (selectedPredictPaths.length === 0) return;
    const confirmBulk = window.confirm(`Move/copy ${selectedPredictPaths.length} selected images to Class folder?`);
    if (!confirmBulk) return;
    
    // Process items in loop calling overrides endpoint sequentially
    for (const path of selectedPredictPaths) {
      const item = predictQueue.find(p => p.path === path);
      if (item) {
        await handlePredictOverride(item, action);
      }
    }
    setSelectedPredictPaths([]);
  };

  const exportPredictionCSV = () => {
    if (predictQueue.length === 0) return;
    const rows = [["Path", "Score", "Label"]];
    predictQueue.forEach(item => {
      if (item && item.path) {
        const scoreVal = item.score !== null ? item.score.toFixed(4) : "";
        rows.push([item.path, scoreVal, item.label || ""]);
      }
    });
    
    const csvContent = rows
      .map(e => e.map(val => `"${String(val).replace(/"/g, '""')}"`).join(","))
      .join("\n");
      
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", `${workspace.name.replace(/\s+/g, '_')}_predictions.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  // Automated Pipelines
  const addEmptyBatchJob = () => {
    const defaultDs = getActiveDatasetData();
    const newJob = {
      input: '',
      output: '',
      model: workspace.models[0] || '',
      threshold: 50,
      label_a_name: defaultDs.label_a,
      label_b_name: defaultDs.label_b
    };
    const nextJobs = [...batchJobs, newJob];
    setBatchJobs(nextJobs);
    saveWorkspaceSettings(workspace.settings, nextJobs);
  };

  const removeBatchJob = (index) => {
    const nextJobs = batchJobs.filter((_, idx) => idx !== index);
    setBatchJobs(nextJobs);
    saveWorkspaceSettings(workspace.settings, nextJobs);
  };

  const updateBatchJob = (index, key, val) => {
    const nextJobs = batchJobs.map((job, idx) => {
      if (idx === index) {
        return { ...job, [key]: val };
      }
      return job;
    });
    setBatchJobs(nextJobs);
    saveWorkspaceSettings(workspace.settings, nextJobs);
  };

  const runAllBatchJobs = async () => {
    if (isBatchRunning) return;
    setIsBatchRunning(true);
    setBatchProgress(0);
    setBatchStatusText('Running automated jobs...');
    setBatchLog([]);
    
    try {
      const res = await fetch(`${API_BASE}/api/workspace/batch/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scripts: batchJobs })
      });
      const data = await res.json();
      if (data.error) {
        showCustomAlert(data.error);
        setIsBatchRunning(false);
      }
    } catch (e) {
      console.error(e);
      setIsBatchRunning(false);
    }
  };

  const stopBatchJobs = async () => {
    try {
      await fetch(`${API_BASE}/api/workspace/batch/stop`, { method: 'POST' });
    } catch (e) {
      console.error(e);
    }
  };

  // Pipelines Poller
  useEffect(() => {
    let timer;
    if (isBatchRunning) {
      timer = setInterval(async () => {
        try {
          const res = await fetch(`${API_BASE}/api/workspace/batch/status`);
          const data = await res.json();
          setBatchProgress(data.progress || 0);
          setBatchLog(data.log || []);
          
          if (data.status !== 'running') {
            setIsBatchRunning(false);
            clearInterval(timer);
            setBatchStatusText(data.status === 'completed' ? 'Automated batch completed.' : 'Batch process stopped.');
            checkActiveWorkspace();
          } else {
            setBatchStatusText('Running automated pipelines...');
          }
        } catch (e) {
          console.error(e);
        }
      }, 1000);
    }
    return () => clearInterval(timer);
  }, [isBatchRunning]);

  if (loading) {
    return (
      <div className="flex align-center justify-between w-full h-full" style={{ justifyContent: 'center', height: '100vh', flexDirection: 'column', gap: 16 }}>
        <RefreshCw className="animate-spin text-primary" size={40} />
        <p style={{ color: 'var(--muted)', fontWeight: 700, fontSize: '0.9rem' }}>Initializing GPU backends...</p>
      </div>
    );
  }

  // Welcome selection layout
  if (!workspace) {
    return (
      <div className="workspace-scroller" style={{ overflowY: 'auto', height: '100vh' }}>
        <div className="workspace-wrapper" style={{ display: 'flex', alignItems: 'center', minHeight: '100vh' }}>
          <div className="welcome-hub">
            <div className="welcome-logo">
              <Layers size={32} />
            </div>
            <h2 className="welcome-title-text">BiLabel Image Tagger Studio</h2>
            <p className="welcome-subtitle-text">Redesigned lightweight binary classification studio utilizing CLIP feature extractions.</p>
            
            {errorMsg && (
              <div style={{ border: '1px solid var(--danger)', backgroundColor: 'var(--danger-glow)', display: 'flex', alignItems: 'center', gap: 12, padding: '12px 18px', marginBottom: 24, borderRadius: 8 }}>
                <AlertTriangle className="text-danger" size={20} />
                <p style={{ color: 'var(--text-dark)', fontSize: '0.85rem', textAlign: 'left', fontWeight: 600 }}>{errorMsg}</p>
              </div>
            )}
  
            <div className="welcome-actions-row">
              <div className="welcome-action-box" onClick={() => setShowCreateModal(true)}>
                <div className="welcome-action-box-icon create"><Plus size={20} /></div>
                <h3>Create Workspace</h3>
                <p>Initialize a new workspace configuration folder structure on your local storage.</p>
              </div>
              <div className="welcome-action-box" onClick={() => setShowOpenModal(true)}>
                <div className="welcome-action-box-icon open"><Folder size={20} /></div>
                <h3>Open Workspace</h3>
                <p>Import and reload an existing configuration JSON layout from a local folder.</p>
              </div>
            </div>
  
            {recents.length > 0 && (
              <div className="unsloth-card">
                <h3 className="unsloth-card-header" style={{ fontSize: '0.95rem', fontWeight: 700, margin: '0 0 16px 0' }}>
                  <RefreshCw size={14} style={{ marginRight: 6 }} /> Recent Workspace Directories
                </h3>
                <div className="recents-stack">
                  {recents.map((r, idx) => (
                    <div key={idx} className="recent-row" onClick={() => handleOpenWorkspace(r.path)}>
                      <div className="recent-row-title">{r.name}</div>
                      <div className="recent-row-path">{r.path}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
  
          {/* Workspace Modals */}
          {showCreateModal && (
            <div className="dir-modal-overlay">
              <div className="dir-modal-box">
                <div className="dir-modal-header">
                  <h3>Create New Workspace</h3>
                  <button className="unsloth-btn unsloth-btn-secondary" style={{ padding: 4 }} onClick={() => setShowCreateModal(false)}><X size={18} /></button>
                </div>
                <div className="dir-modal-body">
                  <div className="unsloth-field-group">
                    <label>Workspace Folder Name</label>
                    <input type="text" className="unsloth-input" placeholder="e.g. My Anime Classifier" value={newWorkspaceName} onChange={e => setNewWorkspaceName(e.target.value)} />
                  </div>
                  <div className="unsloth-field-group">
                    <label>Parent Folder Destination</label>
                    <div className="flex gap-10">
                      <input type="text" className="unsloth-input" value={newWorkspacePath} readOnly placeholder="Choose parent path location..." />
                      <button className="unsloth-btn unsloth-btn-secondary" onClick={() => openDirectoryBrowser(newWorkspacePath, setNewWorkspacePath)}>Browse</button>
                    </div>
                  </div>
                </div>
                <div className="dir-modal-footer">
                  <button className="unsloth-btn unsloth-btn-secondary" onClick={() => setShowCreateModal(false)}>Cancel</button>
                  <button className="unsloth-btn unsloth-btn-primary" onClick={handleCreateWorkspace}>Create Workspace</button>
                </div>
              </div>
            </div>
          )}
  
          {/* Workspace Modals */}
          {showOpenModal && (
            <div className="dir-modal-overlay">
              <div className="dir-modal-box">
                <div className="dir-modal-header">
                  <h3>Open Existing Workspace</h3>
                  <button className="unsloth-btn unsloth-btn-secondary" style={{ padding: 4 }} onClick={() => setShowOpenModal(false)}><X size={18} /></button>
                </div>
                <div className="dir-modal-body">
                  <div className="unsloth-field-group">
                    <label>Workspace Folder Path</label>
                    <div className="flex gap-10">
                      <input type="text" className="unsloth-input" value={openWorkspacePath} readOnly placeholder="Select folder containing workspace.json..." />
                      <button className="unsloth-btn unsloth-btn-secondary" onClick={() => openDirectoryBrowser(openWorkspacePath, setOpenWorkspacePath)}>Browse</button>
                    </div>
                  </div>
                </div>
                <div className="dir-modal-footer">
                  <button className="unsloth-btn unsloth-btn-secondary" onClick={() => setShowOpenModal(false)}>Cancel</button>
                  <button className="unsloth-btn unsloth-btn-primary" onClick={() => handleOpenWorkspace(openWorkspacePath)}>Load Workspace</button>
                </div>
              </div>
            </div>
          )}
  
          {/* File Browser Modal */}
          {showBrowser && renderFolderBrowser()}
        </div>
      </div>
    );
  }

  // Active Workspace Navigation Tabs
  return (
    <div className="app-container">
      <header className="top-header">
        <div className="logo-section">
          <div className="logo-icon-svg"><Layers size={18} /></div>
          <span className="logo-text-title">PrefNet Studio <span className="logo-tag">CLIP & NN</span></span>
        </div>

        {/* 6 Tab Capsule */}
        <div className="nav-capsule">
          <button className={`nav-pill ${activeTab === 'dashboard' ? 'active' : ''}`} onClick={() => setActiveTab('dashboard')}>
            Dashboard
          </button>
          <button className={`nav-pill ${activeTab === 'dataset' ? 'active' : ''}`} onClick={() => setActiveTab('dataset')}>
            Dataset Studio
          </button>
          <button className={`nav-pill ${activeTab === 'labeler' ? 'active' : ''}`} onClick={() => setActiveTab('labeler')}>
            Labeler
          </button>
          <button className={`nav-pill ${activeTab === 'trainer' ? 'active' : ''}`} onClick={() => setActiveTab('trainer')}>
            Trainer
          </button>
          <button className={`nav-pill ${activeTab === 'predictions' ? 'active' : ''}`} onClick={() => setActiveTab('predictions')}>
            Predictions
          </button>
          <button className={`nav-pill ${activeTab === 'pipelines' ? 'active' : ''}`} onClick={() => setActiveTab('pipelines')}>
            Pipelines
          </button>
        </div>

        <div className="header-right">
          {updateAvailable && (
            <button 
              className="unsloth-btn"
              style={{
                backgroundColor: 'var(--success)',
                color: 'white',
                border: 'none',
                padding: '6px 12px',
                fontSize: '0.75rem',
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                fontWeight: 'bold',
                cursor: 'pointer',
                borderRadius: '6px'
              }}
              onClick={() => window.open("https://github.com/Dominik7272/PrefNet", "_blank")}
            >
              <Download size={12} /> Update Available ({latestVersion})
            </button>
          )}
          <span className="flex align-center gap-6" style={{ color: 'var(--primary)' }}><Cpu size={14} /> device: {workspace.device.toUpperCase()}</span>
          <span>Workspace: <b>{workspace.name}</b></span>
          <button className="unsloth-btn unsloth-btn-secondary" style={{ padding: '6px 12px', fontSize: '0.75rem' }} onClick={() => setWorkspace(null)}>
            Close
          </button>
        </div>
      </header>

      {/* Main Container */}
      <div className="workspace-scroller">
        <div className="workspace-wrapper">
          {activeTab === 'dashboard' && renderDashboardTab()}
          {activeTab === 'dataset' && renderDatasetTab()}
          {activeTab === 'labeler' && renderLabelerTab()}
          {activeTab === 'trainer' && renderTrainerTab()}
          {activeTab === 'predictions' && renderPredictionsTab()}
          {activeTab === 'pipelines' && renderPipelinesTab()}
        </div>
      </div>

      {/* Folders Browser modal */}
      {showBrowser && renderFolderBrowser()}

      {/* Custom Alert Modal */}
      {customAlert.show && (
        <div className="dir-modal-overlay" style={{ zIndex: 9999 }}>
          <div className="dir-modal-box" style={{ maxWidth: 400, border: '1px solid var(--danger)', backgroundColor: '#ffffff' }}>
            <div className="dir-modal-header" style={{ borderBottom: '1px solid var(--border)' }}>
              <h3 style={{ color: 'var(--danger)', display: 'flex', alignItems: 'center', gap: 8 }}>
                <AlertTriangle size={18} /> {customAlert.title}
              </h3>
              <button className="unsloth-btn unsloth-btn-secondary" style={{ padding: 4 }} onClick={() => setCustomAlert({ ...customAlert, show: false })}><X size={18} /></button>
            </div>
            <div className="dir-modal-body" style={{ padding: '20px 24px' }}>
              <p style={{ color: 'var(--text-dark)', fontSize: '0.85rem', fontWeight: 500, lineHeight: 1.5 }}>
                {customAlert.message}
              </p>
            </div>
            <div className="dir-modal-footer" style={{ borderTop: 'none', padding: '12px 24px 20px' }}>
              <button className="unsloth-btn w-full" style={{ backgroundColor: 'var(--danger)', color: 'white' }} onClick={() => setCustomAlert({ ...customAlert, show: false })}>
                Acknowledge
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );

  // TAB 1: Dashboard Overview
  function renderDashboardTab() {
    const actStats = workspace.stats[workspace.settings.active_dataset] || { count_a: 0, count_b: 0, path: '' };
    const totalLabeled = Object.values(workspace.stats).reduce((acc, cur) => acc + (cur.count_a || 0) + (cur.count_b || 0), 0);
    const modelsCount = workspace.models.length;

    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
        <div className="studio-section-header">
          <h2>Dashboard Overview</h2>
          <p>Control center containing active dataset profiles, system configurations, and model metrics.</p>
        </div>

        {/* 4 Summary Cards */}
        <div className="summary-cards-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))' }}>
          <div className="summary-card">
            <div className="summary-card-icon primary"><Layers size={22} /></div>
            <div className="summary-card-info">
              <h5>Active Dataset</h5>
              <p style={{ fontSize: '1.1rem', fontWeight: 700 }}>{workspace.settings.active_dataset}</p>
            </div>
          </div>
          <div className="summary-card">
            <div className="summary-card-icon success"><Check size={22} /></div>
            <div className="summary-card-info">
              <h5>Total Labeled (All Profiles)</h5>
              <p style={{ fontSize: '1.4rem', fontWeight: 800 }}>{totalLabeled}</p>
            </div>
          </div>
          <div className="summary-card">
            <div className="summary-card-icon purple"><Sliders size={22} /></div>
            <div className="summary-card-info">
              <h5>Trained weights files</h5>
              <p style={{ fontSize: '1.4rem', fontWeight: 800 }}>{modelsCount}</p>
            </div>
          </div>
          <div className="summary-card">
            <div className="summary-card-icon amber"><Cpu size={22} /></div>
            <div className="summary-card-info">
              <h5>Backend hardware</h5>
              <p style={{ fontSize: '1.1rem', fontWeight: 700, textTransform: 'uppercase' }}>{workspace.device}</p>
            </div>
          </div>
        </div>

        {/* Mid Row: Datasets Table & Models Table */}
        <div className="unsloth-workspace-grid-split">
          {/* Dataset profiles detailed */}
          <div className="unsloth-card">
            <h3 className="unsloth-card-header" style={{ fontSize: '0.95rem', margin: '0 0 16px 0' }}><Database size={16} style={{ marginRight: 6 }} /> Dataset Profiles Summary</h3>
            <div style={{ maxHeight: 220, overflowY: 'auto' }}>
              <table className="model-table">
                <thead>
                  <tr>
                    <th>Profile Name</th>
                    <th>{activeDs.label_a} (A)</th>
                    <th>{activeDs.label_b} (B)</th>
                    <th>Total</th>
                    <th>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {workspace.settings.datasets.map(ds => {
                    const stats = workspace.stats[ds.name] || { count_a: 0, count_b: 0 };
                    const isActive = workspace.settings.active_dataset === ds.name;
                    return (
                      <tr key={ds.name} style={{ backgroundColor: isActive ? 'var(--primary-glow)' : 'transparent' }}>
                        <td style={{ fontWeight: isActive ? 700 : 500, color: 'var(--text-dark)' }}>{ds.name} {isActive && '(Active)'}</td>
                        <td>{stats.count_a}</td>
                        <td>{stats.count_b}</td>
                        <td style={{ fontWeight: 700 }}>{stats.count_a + stats.count_b}</td>
                        <td>
                          {isActive ? (
                            <span style={{ fontSize: '0.75rem', color: 'var(--primary)', fontWeight: 700 }}>Activated</span>
                          ) : (
                            <button className="unsloth-btn unsloth-btn-secondary" style={{ padding: '2px 8px', fontSize: '0.7rem' }} onClick={() => activateDatasetProfile(ds.name)}>
                              Activate
                            </button>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Model Weights List */}
          <div className="unsloth-card">
            <h3 className="unsloth-card-header" style={{ fontSize: '0.95rem', margin: '0 0 16px 0' }}><Sliders size={16} style={{ marginRight: 6 }} /> Model weights pth list</h3>
            {workspace.models_detailed && workspace.models_detailed.length > 0 ? (
              <div style={{ maxHeight: 220, overflowY: 'auto' }}>
                <table className="model-table">
                  <thead>
                    <tr>
                      <th>Weight Name</th>
                      <th>Size (MB)</th>
                      <th>Last Modified</th>
                    </tr>
                  </thead>
                  <tbody>
                    {workspace.models_detailed.map(m => (
                      <tr key={m.name}>
                        <td style={{ color: 'var(--text-dark)' }}>{m.name}</td>
                        <td>{m.size_mb} MB</td>
                        <td>{m.modified}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p style={{ color: 'var(--muted)', fontSize: '0.85rem', textAlign: 'center', padding: 40 }}>No trained weights models located. Proceed to Trainer tab.</p>
            )}
          </div>
        </div>

        {/* Bottom Row: System details & Quick Start */}
        <div className="unsloth-workspace-grid-split">
          {/* Workspace meta details */}
          <div className="unsloth-card">
            <h3 className="unsloth-card-header" style={{ fontSize: '0.95rem', margin: '0 0 16px 0' }}><Settings size={16} style={{ marginRight: 6 }} /> Workspace config info</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12, fontSize: '0.85rem' }}>
              <div><b>Workspace Name:</b> {workspace.name}</div>
              <div><b>Database Location:</b> <span style={{ fontFamily: 'monospace', fontSize: '0.75rem', color: 'var(--muted)' }}>{workspace.path}</span></div>
              <div><b>CLIP model config ID:</b> <span style={{ fontFamily: 'monospace', fontSize: '0.75rem', color: 'var(--muted)' }}>{workspace.settings.clip_model_id}</span></div>
              <div><b>Decision Split Threshold:</b> {workspace.settings.threshold}%</div>
              <div><b>COPY/MOVE Organization mode:</b> Files are organized using system <span style={{ textTransform: 'uppercase', fontWeight: 'bold', color: 'var(--primary)' }}>{workspace.settings.copy_move_mode}</span> operations.</div>
            </div>
          </div>

          {/* Quick jump actions links */}
          <div className="unsloth-card" style={{ display: 'flex', flexDirection: 'column' }}>
            <h3 className="unsloth-card-header" style={{ fontSize: '0.95rem', margin: '0 0 16px 0' }}><Play size={16} style={{ marginRight: 6 }} /> Quick start pipelines</h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
              <button className="unsloth-btn unsloth-btn-secondary" style={{ padding: '12px', fontSize: '0.8rem', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6 }} onClick={() => setActiveTab('labeler')}>
                <CheckCircle size={20} className="text-success" />
                <b>Interactive Labeler</b>
                <span style={{ fontSize: '0.65rem', color: 'var(--muted)' }}>Label images queue</span>
              </button>
              <button className="unsloth-btn unsloth-btn-secondary" style={{ padding: '12px', fontSize: '0.8rem', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6 }} onClick={() => setActiveTab('trainer')}>
                <Activity size={20} className="text-primary" />
                <b>Train Classifier</b>
                <span style={{ fontSize: '0.65rem', color: 'var(--muted)' }}>Optimize weights NN</span>
              </button>
              <button className="unsloth-btn unsloth-btn-secondary" style={{ padding: '12px', fontSize: '0.8rem', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6 }} onClick={() => setActiveTab('predictions')}>
                <Grid size={20} className="text-purple" />
                <b>Bulk Predictions</b>
                <span style={{ fontSize: '0.65rem', color: 'var(--muted)' }}>Run inference models</span>
              </button>
              <button className="unsloth-btn unsloth-btn-secondary" style={{ padding: '12px', fontSize: '0.8rem', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6 }} onClick={() => setActiveTab('dataset')}>
                <Database size={20} className="text-amber" />
                <b>Dataset Studio</b>
                <span style={{ fontSize: '0.65rem', color: 'var(--muted)' }}>Manage directories profile</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // TAB 2: Dataset Studio
  function renderDatasetTab() {
    return (
      <div>
        <div className="studio-section-header">
          <h2>Dataset Studio</h2>
          <p>Configure dataset classes profiles, modify label titles, and audit labeled collections.</p>
        </div>

        {/* Column layout: configure profiles & check gallery */}
        <div className="unsloth-workspace-grid-split">
          {/* Left: list of profiles */}
          <div>
            <div className="unsloth-card">
              <h3 className="unsloth-card-header" style={{ fontSize: '0.95rem', margin: '0 0 16px 0' }}><Plus size={16} style={{ marginRight: 6 }} /> Create Dataset profile</h3>
              <div className="unsloth-field-group">
                <label>Profile Name</label>
                <input type="text" className="unsloth-input" placeholder="e.g. Wallpaper Tagger" value={dsName} onChange={e => setDsName(e.target.value)} />
              </div>
              <div className="unsloth-grid-row">
                <div className="unsloth-field-group">
                  <label>Class A Label (Positive)</label>
                  <input type="text" className="unsloth-input" placeholder="e.g. Good" value={dsLabelA} onChange={e => setDsLabelA(e.target.value)} />
                </div>
                <div className="unsloth-field-group">
                  <label>Class B Label (Negative)</label>
                  <input type="text" className="unsloth-input" placeholder="e.g. Bad" value={dsLabelB} onChange={e => setDsLabelB(e.target.value)} />
                </div>
              </div>
              <button className="unsloth-btn unsloth-btn-primary w-full mt-12" onClick={addDatasetProfile}>
                Add Dataset Profile
              </button>
            </div>

            <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '1.05rem', fontWeight: 800, marginBottom: 12 }}>Dataset profiles</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {workspace.settings.datasets.map(ds => {
                const isActive = workspace.settings.active_dataset === ds.name;
                const stats = workspace.stats[ds.name] || { count_a: 0, count_b: 0, path: '' };
                return (
                  <DatasetRowCard 
                    key={ds.name} 
                    ds={ds} 
                    isActive={isActive} 
                    stats={stats} 
                    onActivate={() => activateDatasetProfile(ds.name)}
                    onDelete={() => deleteDatasetProfile(ds.name)}
                    onSave={(name, lA, lB) => saveDatasetProfileChanges(ds.name, name, lA, lB)}
                  />
                );
              })}
            </div>
          </div>

          {/* Right: Visual auditing gallery */}
          <div className="unsloth-card" style={{ display: 'flex', flexDirection: 'column' }}>
            <h3 className="unsloth-card-header" style={{ fontSize: '0.95rem', margin: '0 0 16px 0' }}><Eye size={16} style={{ marginRight: 6 }} /> Audit Labeled Collections</h3>
            <p style={{ color: 'var(--muted)', fontSize: '0.8rem', marginBottom: 16 }}>Audit and clean up images in your active dataset.</p>

            <div className="flex gap-6" style={{ marginBottom: 12 }}>
              <button className={`unsloth-btn unsloth-btn-secondary ${auditClass === 'a' ? 'active' : ''}`} style={auditClass === 'a' ? { backgroundColor: 'var(--success-glow)', borderColor: 'var(--success)', color: 'var(--success)' } : {}} onClick={() => fetchAuditGallery('a')}>
                {activeDs.label_a} Images ({workspace.stats[workspace.settings.active_dataset]?.count_a || 0})
              </button>
              <button className={`unsloth-btn unsloth-btn-secondary ${auditClass === 'b' ? 'active' : ''}`} style={auditClass === 'b' ? { backgroundColor: 'var(--danger-glow)', borderColor: 'var(--danger)', color: 'var(--danger)' } : {}} onClick={() => fetchAuditGallery('b')}>
                {activeDs.label_b} Images ({workspace.stats[workspace.settings.active_dataset]?.count_b || 0})
              </button>
            </div>

            {loadingAudit ? (
              <div className="flex align-center justify-between w-full" style={{ justifyContent: 'center', height: 160 }}>
                <RefreshCw className="animate-spin text-primary" size={24} />
              </div>
            ) : (
              <div>
                <div className="gallery-grid">
                  {auditImages.map((p, idx) => (
                    <div key={idx} className="gallery-card">
                      <img src={`${API_BASE}/api/fs/image?path=${encodeURIComponent(p)}`} alt="audit-thumb" loading="lazy" />
                      <div className="gallery-card-overlay">
                        <button className="unsloth-btn" style={{ padding: '6px 8px', backgroundColor: 'var(--primary)', color: 'white' }} title="Move to opposite category" onClick={() => handleAuditAction(p, 'move')}>
                          <ArrowUpDown size={12} />
                        </button>
                        <button className="unsloth-btn" style={{ padding: '6px 8px', backgroundColor: 'var(--danger)', color: 'white' }} title="Delete from dataset" onClick={() => handleAuditAction(p, 'delete')}>
                          <Trash size={12} />
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
                {auditImages.length === 0 && (
                  <p style={{ color: 'var(--muted)', fontSize: '0.8rem', textAlign: 'center', marginTop: 40 }}>No labeled images found in this class folder directory.</p>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  // Row item card for Dataset list
  function DatasetRowCard({ ds, isActive, stats, onActivate, onDelete, onSave }) {
    const [name, setName] = useState(ds.name);
    const [labelA, setLabelA] = useState(ds.label_a);
    const [labelB, setLabelB] = useState(ds.label_b);
    const [isEditing, setIsEditing] = useState(false);

    return (
      <div className="unsloth-card" style={isActive ? { borderColor: 'var(--primary)', boxShadow: '0 4px 15px var(--primary-glow)' } : { padding: 18, marginBottom: 12 }}>
        <div className="flex justify-between align-center" style={{ marginBottom: 12 }}>
          <div className="flex align-center">
            <h4 style={{ fontWeight: 800, fontSize: '0.95rem', color: 'var(--text-dark)' }}>{ds.name}</h4>
            {isActive && <span className="unsloth-badge primary">Active</span>}
          </div>
          <div className="flex gap-6">
            {!isActive && (
              <button className="unsloth-btn unsloth-btn-secondary" style={{ padding: '4px 8px', fontSize: '0.75rem' }} onClick={onActivate}>
                Activate
              </button>
            )}
            {isEditing ? (
              <>
                <button className="unsloth-btn" style={{ padding: '4px 8px', fontSize: '0.75rem', backgroundColor: 'var(--success)', color: 'white' }} onClick={() => { onSave(name, labelA, labelB); setIsEditing(false); }}>
                  Save
                </button>
                <button className="unsloth-btn unsloth-btn-secondary" style={{ padding: '4px 8px', fontSize: '0.75rem' }} onClick={() => setIsEditing(false)}>
                  Cancel
                </button>
              </>
            ) : (
              <button className="unsloth-btn unsloth-btn-secondary" style={{ padding: '4px 8px', fontSize: '0.75rem' }} onClick={() => setIsEditing(true)}>
                Rename labels
              </button>
            )}
            <button className="unsloth-btn unsloth-btn-danger" style={{ padding: '4px 8px', fontSize: '0.75rem' }} disabled={isActive} onClick={onDelete}>
              Delete
            </button>
          </div>
        </div>

        {isEditing ? (
          <div className="unsloth-grid-row">
            <div className="unsloth-field-group" style={{ marginBottom: 0 }}>
              <label>Profile Name</label>
              <input type="text" className="unsloth-input" value={name} onChange={e => setName(e.target.value)} />
            </div>
            <div className="unsloth-field-group" style={{ marginBottom: 0 }}>
              <label>Label A</label>
              <input type="text" className="unsloth-input" value={labelA} onChange={e => setLabelA(e.target.value)} />
            </div>
            <div className="unsloth-field-group" style={{ marginBottom: 0 }}>
              <label>Label B</label>
              <input type="text" className="unsloth-input" value={labelB} onChange={e => setLabelB(e.target.value)} />
            </div>
          </div>
        ) : (
          <div style={{ fontSize: '0.8rem', color: 'var(--muted)', display: 'flex', flexDirection: 'column', gap: 4 }}>
            <div>Class A: <b>{ds.label_a}</b> ({stats.count_a} items)</div>
            <div>Class B: <b>{ds.label_b}</b> ({stats.count_b} items)</div>
            <div style={{ fontFamily: 'monospace', fontSize: '0.7rem', textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }}>
              Path: {stats.path}
            </div>
          </div>
        )}
      </div>
    );
  }

  // TAB 3: Interactive Labeler
  function renderLabelerTab() {
    const queueActive = labelQueue.length > 0;
    const isDone = queueActive && labelIndex >= labelQueue.length;
    const currentImg = queueActive && !isDone ? labelQueue[labelIndex] : null;

    return (
      <div>
        <div className="studio-section-header">
          <h2>Interactive Labeler</h2>
          <p>Watch source image queue, classify elements, and navigate layout using interactive keyboard hotkeys.</p>
        </div>

        {/* Directories selection */}
        <div className="unsloth-card" style={{ padding: '16px 20px' }}>
          <div className="flex align-center gap-16" style={{ flexWrap: 'wrap' }}>
            <div className="flex align-center gap-6" style={{ flexGrow: 1 }}>
              <Folder size={16} className="text-muted" />
              <input type="text" className="unsloth-input" style={{ height: 38 }} readOnly value={labelFolder} placeholder="Choose folder path containing images sources queue..." />
              <button className="unsloth-btn unsloth-btn-secondary" style={{ height: 38 }} onClick={() => openDirectoryBrowser(labelFolder, setLabelFolder, true)}>Browse</button>
            </div>
            <div className="flex align-center gap-10">
              <label style={{ fontSize: '0.8rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer' }}>
                <input type="checkbox" checked={labelRecursive} onChange={e => setLabelRecursive(e.target.checked)} />
                Recursive Subfolders
              </label>
              <button className="unsloth-btn unsloth-btn-primary" disabled={!labelFolder} onClick={loadLabelQueue}>
                Load images queue
              </button>
            </div>
          </div>
        </div>

        {queueActive ? (
          <div className="labeling-layout">
            
            {/* Viewport block with Panning & Zooming */}
            <div 
              className="stage-wrapper-unsloth" 
              ref={labelViewportRef}
              onMouseDown={handlePanStart}
              onMouseMove={handlePanning}
              onMouseUp={handlePanEnd}
              onMouseLeave={handlePanEnd}
              onDoubleClick={resetViewportZoom}
              style={{ cursor: isPanning ? 'grabbing' : 'grab' }}
            >
              {currentImg ? (
                <>
                  <img 
                    src={`${API_BASE}/api/fs/image?path=${encodeURIComponent(currentImg)}`} 
                    alt="label-target" 
                    style={{
                      transform: `translate(${panOffset.x}px, ${panOffset.y}px) scale(${zoomScale})`,
                      transformOrigin: 'center center',
                      transition: isPanning ? 'none' : 'transform 0.15s cubic-bezier(0.1, 0.8, 0.3, 1)'
                    }}
                  />
                  <div className="hud-counter">
                    Queue Position: {labelIndex + 1} / {labelQueue.length}
                  </div>
                  <div className="hud-instructions">
                    Hotkeys: <kbd>→</kbd> {activeDs.label_a} | <kbd>←</kbd> {activeDs.label_b} | <kbd>↑</kbd> Skip | <kbd>↓</kbd> Undo last | <kbd>Scroll</kbd> Zoom | <kbd>Double-Click</kbd> Reset view
                  </div>
                </>
              ) : (
                <div style={{ textAlign: 'center', padding: 40, color: 'var(--muted)' }}>
                  <CheckSquare size={36} style={{ marginBottom: 12, color: 'var(--primary)' }} />
                  <h4 style={{ color: 'var(--text-dark)', fontWeight: 800 }}>Queue Completed!</h4>
                  <p style={{ fontSize: '0.85rem', marginTop: 4 }}>Select another folder location to import more unlabelled collections.</p>
                </div>
              )}
            </div>

            {/* Sidebar actions & Labeled strip */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
              <div className="unsloth-card" style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                <div className="flex justify-between align-center" style={{ marginBottom: 6 }}>
                  <h4 style={{ fontSize: '0.9rem', fontWeight: 700, color: 'var(--text-dark)' }}>Category Action Controls</h4>
                  <button className="unsloth-btn unsloth-btn-secondary" style={{ padding: '4px 8px', fontSize: '0.7rem' }} onClick={resetViewportZoom}>
                    Reset Zoom/Pan
                  </button>
                </div>

                {(() => {
                  const previousChoiceObj = labelHistory.find(h => h.index === labelIndex);
                  const previousChoice = previousChoiceObj ? previousChoiceObj.action : null;
                  return (
                    <>
                      <button 
                        className="unsloth-btn w-full" 
                        style={{ 
                          backgroundColor: 'var(--success)', 
                          color: 'white',
                          border: previousChoice === 'a' ? '3px solid var(--text-dark)' : 'none',
                          opacity: previousChoice && previousChoice !== 'a' ? 0.65 : 1
                        }} 
                        disabled={isDone} 
                        onClick={() => executeLabelAction('a')}
                      >
                        {previousChoice === 'a' ? '✓ ' : ''}Save as A: {activeDs.label_a} <ArrowRight size={14} />
                      </button>
                      
                      <button 
                        className="unsloth-btn w-full" 
                        style={{ 
                          backgroundColor: 'var(--danger)', 
                          color: 'white',
                          border: previousChoice === 'b' ? '3px solid var(--text-dark)' : 'none',
                          opacity: previousChoice && previousChoice !== 'b' ? 0.65 : 1
                        }} 
                        disabled={isDone} 
                        onClick={() => executeLabelAction('b')}
                      >
                        <ArrowLeft size={14} /> {previousChoice === 'b' ? '✓ ' : ''}Save as B: {activeDs.label_b}
                      </button>

                      <button 
                        className="unsloth-btn unsloth-btn-secondary w-full" 
                        style={{
                          border: previousChoice === 'skip' ? '3px solid var(--text-dark)' : 'none',
                          opacity: previousChoice && previousChoice !== 'skip' ? 0.65 : 1
                        }}
                        disabled={isDone} 
                        onClick={() => executeLabelAction('skip')}
                      >
                        {previousChoice === 'skip' ? '✓ ' : ''}Skip file <SkipForward size={14} />
                      </button>
                    </>
                  );
                })()}

                <button className="unsloth-btn unsloth-btn-secondary w-full" disabled={labelHistory.length === 0} onClick={undoLabelAction}>
                  Undo last move <RotateCcw size={14} />
                </button>
              </div>

              {/* Labeled image strip (History Thumbnail strip) */}
              <div className="unsloth-card" style={{ flexGrow: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', minHeight: 200 }}>
                <h4 style={{ fontSize: '0.9rem', fontWeight: 700, color: 'var(--text-dark)', marginBottom: 8 }}>Session strip timeline</h4>
                <p style={{ color: 'var(--muted)', fontSize: '0.75rem', marginBottom: 12 }}>Click any thumbnail to jump to that image in the queue.</p>
                <div style={{ display: 'flex', gap: 8, overflowX: 'auto', paddingBottom: 10, flexWrap: 'wrap' }}>
                  {labelHistory.map((h, idx) => (
                    <div 
                      key={idx} 
                      onClick={() => setLabelIndex(h.index)}
                      style={{ 
                        width: 50, 
                        height: 50, 
                        border: '2px solid transparent', 
                        borderColor: h.action === 'skip' ? '#9ca3af' : h.action === 'a' ? 'var(--success)' : 'var(--danger)',
                        borderRadius: 6,
                        overflow: 'hidden',
                        cursor: 'pointer',
                        position: 'relative'
                      }}
                      title={`Revert to: ${h.src.split(/[\\/]/).pop()}`}
                    >
                      <img src={`${API_BASE}/api/fs/image?path=${encodeURIComponent(h.src)}`} alt="strip-thumb" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                    </div>
                  ))}
                  {labelHistory.length === 0 && (
                    <p style={{ color: 'var(--muted)', fontSize: '0.8rem', textAlign: 'center', width: '100%', marginTop: 32 }}>No history strip logged yet.</p>
                  )}
                </div>
              </div>
            </div>

          </div>
        ) : (
          <div className="unsloth-card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyItems: 'center', padding: '60px 20px', color: 'var(--muted)' }}>
            <Database size={40} style={{ marginBottom: 12, opacity: 0.6 }} />
            <h4 style={{ color: 'var(--text-dark)', fontWeight: 800 }}>Queue is Empty</h4>
            <p style={{ fontSize: '0.85rem', marginTop: 4 }}>Select target folders and load images collection queue to start classification.</p>
          </div>
        )}
      </div>
    );
  }

  function renderNNBlockBuilder() {
    const selectedLayer = selectedLayerIdx !== null && selectedLayerIdx !== undefined ? customLayers[selectedLayerIdx] : null;



    const renderInsertSeparator = (insertIdx) => {
      const isDragOver = dragOverInsertIdx === insertIdx;
      return (
        <div
          onDragOver={e => {
            e.preventDefault();
            if (dragOverInsertIdx !== insertIdx) setDragOverInsertIdx(insertIdx);
          }}
          onDragLeave={() => {
            setDragOverInsertIdx(null);
          }}
          onDrop={e => {
            e.preventDefault();
            setDragOverInsertIdx(null);
            handleInsertDrop(e, insertIdx);
          }}
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            height: 24,
            margin: '2px 0',
            borderRadius: 4,
            border: isDragOver ? '1.5px dashed var(--success)' : '1.5px solid transparent',
            backgroundColor: isDragOver ? 'var(--success-glow)' : 'transparent',
            transition: 'all 0.15s ease'
          }}
        >
          <ChevronDown size={14} style={{ color: isDragOver ? 'var(--success)' : 'var(--muted)' }} />
        </div>
      );
    };

    const handleLayerCardDrop = (e, targetIdx) => {
      const data = e.dataTransfer.getData('text/plain');
      if (!data) return;
      
      if (data.startsWith('palette:')) {
        const type = data.split(':')[1];
        replaceLayer(targetIdx, type);
      } else if (data.startsWith('layer:')) {
        const srcIdx = parseInt(data.split(':')[1]);
        if (srcIdx !== targetIdx) {
          // Swap layers
          setCustomLayers(prev => {
            const next = [...prev];
            const temp = next[targetIdx];
            next[targetIdx] = next[srcIdx];
            next[srcIdx] = temp;
            saveCustomLayersSettings(next, useCustomNN, null);
            return next;
          });
        }
      }
    };

    const handleInsertDrop = (e, insertIdx) => {
      const data = e.dataTransfer.getData('text/plain');
      if (!data) return;
      
      if (data.startsWith('palette:')) {
        const type = data.split(':')[1];
        insertLayer(insertIdx, type);
      } else if (data.startsWith('layer:')) {
        const srcIdx = parseInt(data.split(':')[1]);
        moveLayer(srcIdx, insertIdx);
      }
    };

    return (
      <div className="unsloth-card" style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
        <div className="flex justify-between align-center" style={{ borderBottom: '1px solid var(--border)', paddingBottom: 12, flexWrap: 'wrap', gap: 10 }}>
          <h3 className="unsloth-card-header" style={{ fontSize: '0.95rem', margin: 0, display: 'flex', alignItems: 'center', gap: 8 }}>
            <Sliders size={16} /> Neural Network Architecture Designer
          </h3>
        </div>

        {/* Preset Selector & Custom Preset Save */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, backgroundColor: 'var(--bg-darker)', padding: 12, borderRadius: 8 }}>
          <div className="unsloth-field-group" style={{ marginBottom: 0 }}>
            <label style={{ fontSize: '0.75rem' }}>Preset Architectures</label>
            <select 
              className="unsloth-input" 
              style={{ height: 36, fontSize: '0.75rem', padding: '6px 10px', lineHeight: 'normal' }} 
              onChange={e => {
                if (!presets || !Array.isArray(presets)) return;
                const selected = presets.find(p => p && p.name === e.target.value);
                if (selected) {
                  setCustomLayers(selected.layers);
                  saveCustomLayersSettings(selected.layers, useCustomNN, null);
                  setSelectedLayerIdx(0);
                }
              }}
            >
              <option value="">-- Apply Preset --</option>
              {presets && Array.isArray(presets) && presets.map((p, pIdx) => {
                if (!p || !p.name) return null;
                return (
                  <option key={pIdx} value={p.name}>{p.name}</option>
                );
              })}
            </select>
          </div>
          <div className="unsloth-field-group" style={{ marginBottom: 0 }}>
            <label style={{ fontSize: '0.75rem' }}>Save Preset</label>
            <div className="flex gap-6">
              <input 
                type="text" 
                className="unsloth-input" 
                style={{ height: 36, fontSize: '0.75rem', padding: '6px 10px' }} 
                placeholder="Custom Preset Name" 
                value={newPresetName} 
                onChange={e => setNewPresetName(e.target.value)} 
              />
              <button className="unsloth-btn unsloth-btn-primary" style={{ padding: '0 12px', height: 36, fontSize: '0.75rem' }} onClick={handleSavePreset}>
                Save
              </button>
            </div>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '220px 1fr 280px', gap: 16 }}>
            
            {/* LEFT COLUMN: Available Blocks Palette */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14, borderRight: '1px solid var(--border)', paddingRight: 12 }}>
              <div style={{ paddingBottom: 6 }}>
                <h4 style={{ fontSize: '0.9rem', fontWeight: 800, color: 'var(--text-dark)', margin: 0 }}>Layers</h4>
                <span style={{ fontSize: '0.7rem', color: 'var(--muted)', display: 'block', marginTop: 2 }}>Drag layers to add/replace</span>
              </div>
              
              <span style={{ display: 'block', fontSize: '0.65rem', color: 'var(--muted)', fontWeight: 800, letterSpacing: '0.5px', textTransform: 'uppercase' }}>
                Basic Layers
              </span>
              
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {[
                  { type: 'Linear', name: 'Linear (Dense)', color: 'var(--primary)', glow: 'var(--primary-glow)' },
                  { type: 'BatchNorm1d', name: 'BatchNorm 1D', color: 'var(--blue)', glow: 'var(--blue-glow)' },
                  { type: 'Dropout', name: 'Dropout Rate', color: 'var(--purple)', glow: 'var(--purple-glow)' },
                  { type: 'ReLU', name: 'ReLU Activation', color: 'var(--amber)', glow: 'var(--amber-glow)' },
                  { type: 'LeakyReLU', name: 'LeakyReLU Activation', color: 'var(--amber)', glow: 'var(--amber-glow)' },
                  { type: 'Sigmoid', name: 'Sigmoid Activation', color: 'var(--amber)', glow: 'var(--amber-glow)' },
                  { type: 'Tanh', name: 'Tanh Activation', color: 'var(--amber)', glow: 'var(--amber-glow)' }
                ].map(block => (
                  <div
                    key={block.type}
                    draggable="true"
                    onDragStart={e => {
                      e.dataTransfer.effectAllowed = 'copy';
                      e.dataTransfer.setData('text/plain', `palette:${block.type}`);
                    }}
                    onClick={() => {
                      setCustomLayers(prev => {
                        const next = [...prev, { type: block.type }];
                        saveCustomLayersSettings(next, useCustomNN, null);
                        return next;
                      });
                      setSelectedLayerIdx(customLayers.length);
                    }}
                    style={{
                      cursor: 'grab',
                      padding: '8px 12px',
                      borderRadius: 6,
                      border: `1.5px dashed ${block.color}`,
                      color: block.color,
                      backgroundColor: block.glow,
                      fontSize: '0.75rem',
                      fontWeight: 700,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      transition: 'all 0.15s ease'
                    }}
                    onMouseEnter={e => {
                      e.currentTarget.style.transform = 'translateY(-1px)';
                      e.currentTarget.style.boxShadow = `0 4px 10px ${block.glow}`;
                    }}
                    onMouseLeave={e => {
                      e.currentTarget.style.transform = 'none';
                      e.currentTarget.style.boxShadow = 'none';
                    }}
                  >
                    <span>+ {block.name}</span>
                    <span style={{ fontSize: '0.65rem', opacity: 0.7, textTransform: 'uppercase' }}>DRAG</span>
                  </div>
                ))}
              </div>
            </div>

            {/* MIDDLE COLUMN: Simple Vertical Blocks List (Light Mode Compatible) */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              <span style={{ display: 'block', fontSize: '0.7rem', color: 'var(--muted)', fontWeight: 800, letterSpacing: '0.5px', textTransform: 'uppercase' }}>
                Network Architecture Stack
              </span>
              
              <div style={{ 
                display: 'flex', 
                flexDirection: 'column', 
                backgroundColor: 'var(--bg-darker)', 
                padding: 16, 
                borderRadius: 8, 
                border: '1px solid var(--border)',
                maxHeight: '520px',
                overflowY: 'auto'
              }}>
                {/* 1. Input Node */}
                <div style={{
                  padding: '10px 14px',
                  borderRadius: 6,
                  border: '1.5px solid var(--blue)',
                  backgroundColor: 'var(--panel)',
                  color: 'var(--text-dark)',
                  fontSize: '0.8rem',
                  fontWeight: 700,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  opacity: 0.85
                }}>
                  <span>CLIP Embeddings (Input)</span>
                  <span style={{ fontSize: '0.65rem', color: 'var(--muted)' }}>
                    {workspace.settings.clip_model_id?.includes('large') ? '768 features' : '512 features'}
                  </span>
                </div>
                
                {/* Down Arrow / Drop Target 0 */}
                {renderInsertSeparator(0)}
                
                {/* 2. Custom Layers */}
                {customLayers.map((layer, idx) => {
                  const isSelected = selectedLayerIdx === idx;
                  const isDragOver = dragOverLayerIdx === idx;
                  
                  let layerColor = 'var(--primary)';
                  if (layer.type === 'BatchNorm1d') layerColor = 'var(--blue)';
                  if (layer.type === 'Dropout') layerColor = 'var(--purple)';
                  if (['ReLU', 'LeakyReLU', 'Sigmoid', 'Tanh'].includes(layer.type)) layerColor = 'var(--amber)';
                  
                  return (
                    <React.Fragment key={idx}>
                      <div
                        draggable="true"
                        onDragStart={e => {
                          e.dataTransfer.effectAllowed = 'move';
                          e.dataTransfer.setData('text/plain', `layer:${idx}`);
                        }}
                        onDragOver={e => {
                          e.preventDefault();
                          if (dragOverLayerIdx !== idx) setDragOverLayerIdx(idx);
                        }}
                        onDragLeave={() => {
                          setDragOverLayerIdx(null);
                        }}
                        onDrop={e => {
                          e.preventDefault();
                          setDragOverLayerIdx(null);
                          handleLayerCardDrop(e, idx);
                        }}
                        onClick={() => setSelectedLayerIdx(idx)}
                        style={{
                          cursor: 'grab',
                          padding: '12px 14px',
                          borderRadius: 6,
                          border: isSelected ? '2px solid var(--primary)' : isDragOver ? '2px dashed var(--amber)' : `1.5px solid var(--border)`,
                          borderLeft: `4px solid ${layerColor}`,
                          backgroundColor: isSelected ? 'var(--panel)' : isDragOver ? 'var(--amber-glow)' : 'var(--panel)',
                          boxShadow: isSelected ? '0 4px 12px var(--primary-glow)' : 'none',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'space-between',
                          transition: 'all 0.15s ease',
                          position: 'relative'
                        }}
                      >
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                          <span style={{ fontSize: '0.8rem', fontWeight: 800, color: 'var(--text-dark)' }}>
                            {idx + 1}. {layer.type}
                          </span>
                          <span style={{ fontSize: '0.7rem', color: 'var(--muted)' }}>
                            {layer.type === 'Linear' && `out_features: ${layer.out_features || 512}${layer.bias ? ', bias' : ''}`}
                            {layer.type === 'BatchNorm1d' && `num_features: ${layer.num_features || 'Auto'}`}
                            {layer.type === 'Dropout' && `rate: ${layer.p !== undefined ? layer.p : 0.2}`}
                            {['ReLU', 'LeakyReLU', 'Sigmoid', 'Tanh'].includes(layer.type) && 'Activation'}
                          </span>
                        </div>
                        
                        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                          <button 
                            className="unsloth-btn unsloth-btn-secondary" 
                            style={{ padding: 4, height: 26, width: 26 }} 
                            onClick={e => {
                              e.stopPropagation();
                              removeLayer(idx);
                            }}
                          >
                            <Trash size={12} className="text-danger" />
                          </button>
                        </div>
                        
                        {/* Drag over replacement visual overlay */}
                        {isDragOver && (
                          <div style={{
                            position: 'absolute',
                            top: 0,
                            left: 0,
                            right: 0,
                            bottom: 0,
                            backgroundColor: 'rgba(249, 115, 22, 0.12)',
                            borderRadius: 6,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            color: 'var(--amber)',
                            fontSize: '0.7rem',
                            fontWeight: 'bold',
                            pointerEvents: 'none'
                          }}>
                            🔄 Drop to Replace
                          </div>
                        )}
                      </div>
                      
                      {/* Insertion point idx + 1 */}
                      {renderInsertSeparator(idx + 1)}
                    </React.Fragment>
                  );
                })}
                
                {/* 3. Output Node */}
                <div style={{
                  padding: '10px 14px',
                  borderRadius: 6,
                  border: '1.5px solid var(--success)',
                  backgroundColor: 'var(--panel)',
                  color: 'var(--text-dark)',
                  fontSize: '0.8rem',
                  fontWeight: 700,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  opacity: 0.85
                }}>
                  <span>Output Prediction (Sigmoid)</span>
                  <span style={{ fontSize: '0.65rem', color: 'var(--muted)' }}>Float rating [0, 1]</span>
                </div>
              </div>
            </div>

            {/* RIGHT COLUMN: Layer Settings Panel */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16, borderLeft: '1px solid var(--border)', paddingLeft: 12 }}>
              {selectedLayer ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                  <div style={{ borderBottom: '1px solid var(--border)', paddingBottom: 10 }}>
                    <span style={{ fontSize: '0.65rem', textTransform: 'uppercase', fontWeight: 800, color: 'var(--primary)', letterSpacing: '0.5px' }}>
                      Block Configuration
                    </span>
                    <h4 style={{ fontSize: '0.9rem', margin: '4px 0 0 0', color: 'var(--text-dark)', fontWeight: 800 }}>
                      Layer {selectedLayerIdx + 1}: {selectedLayer.type} Settings
                    </h4>
                  </div>

                  {selectedLayer.type === 'Linear' && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                      <div className="unsloth-field-group">
                        <label>Output Features</label>
                        <input 
                          type="number" 
                          className="unsloth-input"
                          value={selectedLayer.out_features !== undefined ? selectedLayer.out_features : 512}
                          onChange={e => updateLayerParam(selectedLayerIdx, 'out_features', parseInt(e.target.value) || 1)}
                        />
                        <div className="flex gap-4 mt-12" style={{ flexWrap: 'wrap' }}>
                          {[128, 256, 512, 1024].map(size => (
                            <button
                              key={size}
                              className="unsloth-btn unsloth-btn-secondary"
                              style={{ padding: '4px 8px', fontSize: '0.7rem' }}
                              onClick={() => updateLayerParam(selectedLayerIdx, 'out_features', size)}
                            >
                              {size}
                            </button>
                          ))}
                        </div>
                      </div>

                      <div className="unsloth-field-group">
                        <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', fontWeight: 600 }}>
                          <input 
                            type="checkbox"
                            checked={selectedLayer.bias !== undefined ? selectedLayer.bias : true}
                            onChange={e => updateLayerParam(selectedLayerIdx, 'bias', e.target.checked)}
                          />
                          Enable Linear Bias
                        </label>
                      </div>
                    </div>
                  )}

                  {selectedLayer.type === 'BatchNorm1d' && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                      <div className="unsloth-field-group">
                        <label>Features Size</label>
                        <input 
                          type="number" 
                          className="unsloth-input"
                          placeholder="Leave blank (Auto)"
                          value={selectedLayer.num_features !== undefined ? selectedLayer.num_features : ''}
                          onChange={e => updateLayerParam(selectedLayerIdx, 'num_features', parseInt(e.target.value) || '')}
                        />
                      </div>

                      <div className="unsloth-grid-row" style={{ gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                        <div className="unsloth-field-group">
                          <label>Eps</label>
                          <select 
                            className="unsloth-input"
                            value={selectedLayer.eps !== undefined ? selectedLayer.eps : 1e-5}
                            onChange={e => updateLayerParam(selectedLayerIdx, 'eps', parseFloat(e.target.value))}
                          >
                            <option value="1e-5">1e-5</option>
                            <option value="1e-4">1e-4</option>
                            <option value="1e-6">1e-6</option>
                          </select>
                        </div>
                        <div className="unsloth-field-group">
                          <label>Momentum</label>
                          <select 
                            className="unsloth-input"
                            value={selectedLayer.momentum !== undefined ? selectedLayer.momentum : 0.1}
                            onChange={e => updateLayerParam(selectedLayerIdx, 'momentum', parseFloat(e.target.value))}
                          >
                            <option value="0.1">0.1</option>
                            <option value="0.05">0.05</option>
                            <option value="0.2">0.2</option>
                          </select>
                        </div>
                      </div>
                    </div>
                  )}

                  {selectedLayer.type === 'Dropout' && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                      <div className="unsloth-field-group">
                        <label>Dropout rate (p): {selectedLayer.p !== undefined ? selectedLayer.p : 0.2}</label>
                        <input 
                          type="range"
                          min="0"
                          max="0.9"
                          step="0.05"
                          className="w-full"
                          style={{ accentColor: 'var(--purple)' }}
                          value={selectedLayer.p !== undefined ? selectedLayer.p : 0.2}
                          onChange={e => updateLayerParam(selectedLayerIdx, 'p', parseFloat(e.target.value))}
                        />
                      </div>

                      <div className="unsloth-field-group">
                        <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', fontWeight: 600 }}>
                          <input 
                            type="checkbox"
                            checked={selectedLayer.inplace !== undefined ? selectedLayer.inplace : false}
                            onChange={e => updateLayerParam(selectedLayerIdx, 'inplace', e.target.checked)}
                          />
                          In-place Execution
                        </label>
                      </div>
                    </div>
                  )}

                  {selectedLayer.type === 'ReLU' && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                      <div className="unsloth-field-group">
                        <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', fontWeight: 600 }}>
                          <input 
                            type="checkbox"
                            checked={selectedLayer.inplace !== undefined ? selectedLayer.inplace : false}
                            onChange={e => updateLayerParam(selectedLayerIdx, 'inplace', e.target.checked)}
                          />
                          In-place Execution
                        </label>
                      </div>
                    </div>
                  )}

                  {selectedLayer.type === 'LeakyReLU' && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                      <div className="unsloth-field-group">
                        <label>Negative Slope</label>
                        <input 
                          type="number" 
                          step="0.005"
                          className="unsloth-input"
                          value={selectedLayer.negative_slope !== undefined ? selectedLayer.negative_slope : 0.01}
                          onChange={e => updateLayerParam(selectedLayerIdx, 'negative_slope', parseFloat(e.target.value) || 0.0)}
                        />
                      </div>

                      <div className="unsloth-field-group">
                        <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', fontWeight: 600 }}>
                          <input 
                            type="checkbox"
                            checked={selectedLayer.inplace !== undefined ? selectedLayer.inplace : false}
                            onChange={e => updateLayerParam(selectedLayerIdx, 'inplace', e.target.checked)}
                          />
                          In-place Execution
                        </label>
                      </div>
                    </div>
                  )}

                  {(selectedLayer.type === 'Sigmoid' || selectedLayer.type === 'Tanh') && (
                    <div style={{ backgroundColor: 'var(--panel-hover)', padding: 12, borderRadius: 6, fontSize: '0.75rem', color: 'var(--muted)' }}>
                      No configurable settings for this node type.
                    </div>
                  )}

                  <button
                    className="unsloth-btn unsloth-btn-danger"
                    style={{ display: 'flex', alignItems: 'center', gap: 6, width: 'fit-content', fontSize: '0.8rem', marginTop: 12 }}
                    onClick={() => removeLayer(selectedLayerIdx)}
                  >
                    <Trash size={14} /> Remove Layer
                  </button>
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyItems: 'center', justifyContent: 'center', height: '100%', minHeight: 180, color: 'var(--muted)', border: '1px dashed var(--border)', borderRadius: 8, padding: 16 }}>
                  <Sliders size={24} style={{ marginBottom: 10, opacity: 0.5 }} />
                  <h5 style={{ fontWeight: 700, color: 'var(--text-dark)' }}>No Layer Selected</h5>
                  <p style={{ fontSize: '0.75rem', textAlign: 'center', marginTop: 4, maxWidth: 200 }}>
                    Click on any layer block in the stack to modify its properties here.
                  </p>
                </div>
              )}
            </div>

          </div>
      </div>
    );
  }


  // TAB 4: Trainer NN
  function renderTrainerTab() {
    return (
      <div>
        <div className="studio-section-header">
          <h2>Neural Network Training</h2>
          <p>Extract CLIP image features embeddings and optimize parameters of linear weights layers.</p>
        </div>

        <div className="unsloth-workspace-grid-split" style={{ gridTemplateColumns: '360px 1fr' }}>
          {/* Training Parameter Form */}
          <div className="unsloth-card" style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            <h3 className="unsloth-card-header" style={{ fontSize: '0.95rem', margin: 0 }}><Sliders size={16} /> Parameters optimizer</h3>

            <div className="unsloth-field-group">
              <label>Model weights output Name</label>
              <input type="text" className="unsloth-input" value={trainModelName} onChange={e => setTrainModelName(e.target.value)} disabled={isTraining} />
            </div>

            <div className="unsloth-grid-row" style={{ gridTemplateColumns: '1fr 1fr' }}>
              <div className="unsloth-field-group">
                <label>Epochs</label>
                <input type="number" className="unsloth-input" value={trainEpochs} onChange={e => setTrainEpochs(parseInt(e.target.value) || 20)} disabled={isTraining} />
              </div>
              <div className="unsloth-field-group">
                <label>Batch size</label>
                <input type="number" className="unsloth-input" value={trainBatchSize} onChange={e => setTrainBatchSize(parseInt(e.target.value) || 32)} disabled={isTraining} />
              </div>
            </div>

            <div className="unsloth-field-group">
              <label>Learning Rate (LR)</label>
              <input type="number" step="0.0001" className="unsloth-input" value={trainLearningRate} onChange={e => setTrainLearningRate(parseFloat(e.target.value) || 0.001)} disabled={isTraining} />
            </div>

            <div className="unsloth-field-group">
              <label>CLIP Model Extractor</label>
              <select 
                className="unsloth-input" 
                value={workspace.settings.clip_model_id || 'openai/clip-vit-large-patch14'} 
                onChange={e => {
                  const nextSettings = { ...workspace.settings, clip_model_id: e.target.value };
                  saveWorkspaceSettings(nextSettings);
                }}
                disabled={isTraining}
              >
                <option value="openai/clip-vit-base-patch32">ViT-B/32 (OpenAI) [512d]</option>
                <option value="openai/clip-vit-base-patch16">ViT-B/16 (OpenAI) [512d]</option>
                <option value="openai/clip-vit-large-patch14">ViT-L/14 (OpenAI) [768d]</option>
                <option value="openai/clip-vit-large-patch14-336">ViT-L/14@336px (OpenAI) [768d]</option>
                <option value="laion/CLIP-ViT-H-14-laion2B-s32B-b79K">ViT-H/14 (LAION-2B) [1024d]</option>
                <option value="laion/CLIP-ViT-L-14-laion2B-s32B-b82K">ViT-L/14 (LAION-2B) [768d]</option>
                <option value="laion/CLIP-ViT-B-32-laion2B-s34B-b79K">ViT-B/32 (LAION-2B) [512d]</option>
              </select>
            </div>

            <div className="unsloth-field-group">
              <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', fontWeight: 'bold' }}>
                <input 
                  type="checkbox" 
                  checked={useCustomNN} 
                  onChange={e => toggleUseCustomNN(e.target.checked)} 
                  disabled={isTraining}
                />
                Customize Neural Network
              </label>
            </div>

            <div className="unsloth-field-group">
              <label>Target dataset profile</label>
              <div className="unsloth-input" style={{ backgroundColor: 'var(--bg-darker)', border: 'none' }}>
                {workspace.settings.active_dataset} (Liked vs Disliked)
              </div>
            </div>

            {isTraining ? (
              <button className="unsloth-btn unsloth-btn-danger w-full" onClick={stopModelTraining}>
                <Square size={14} style={{ marginRight: 6 }} /> Stop training process
              </button>
            ) : (
              <button className="unsloth-btn unsloth-btn-primary w-full" onClick={startModelTraining} style={{ background: 'var(--primary)' }}>
                <Play size={14} style={{ marginRight: 6 }} /> Start training optimizer
              </button>
            )}

            {isTraining && (
              <div className="mt-12 w-full">
                <div className="flex justify-between" style={{ fontSize: '0.75rem', marginBottom: 4, fontWeight: 700 }}>
                  <span>Epoch running:</span>
                  <span>{trainProgress}%</span>
                </div>
                <div className="unsloth-progress-track">
                  <div className="unsloth-progress-bar" style={{ width: `${trainProgress}%` }}></div>
                </div>
              </div>
            )}
          </div>

          {/* SVG Plotting Curves & Console Logs output */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
            {/* SVG Loss Curve */}
            <div className="unsloth-card">
              <div className="flex justify-between align-center" style={{ marginBottom: 12 }}>
                <h3 className="unsloth-card-title" style={{ fontSize: '0.95rem', fontWeight: 700, color: 'var(--text-dark)' }}>Live optimizer metrics</h3>
                <div className="flex gap-16" style={{ fontSize: '0.75rem', fontWeight: 700 }}>
                  <span className="flex align-center gap-6"><span style={{ display: 'inline-block', width: 12, height: 6, backgroundColor: 'var(--danger)', borderRadius: 2 }}></span> Loss Value</span>
                  <span className="flex align-center gap-6"><span style={{ display: 'inline-block', width: 12, height: 6, backgroundColor: 'var(--success)', borderRadius: 2 }}></span> Accuracy %</span>
                </div>
              </div>
              <div className="chart-box">
                {renderTrainingChart()}
              </div>
            </div>

            {/* Console Log output */}
            <div className="unsloth-card">
              <h3 className="unsloth-card-header" style={{ fontSize: '0.95rem', margin: 0 }}>Console execution logs</h3>
              <div className="console-box">
                {trainLog.map((line, idx) => (
                  <div key={idx} className="console-row">{line}</div>
                ))}
                {trainLog.length === 0 && <div className="text-muted">Console output logs stream waits for training start...</div>}
                <div ref={logEndRef}></div>
              </div>
            </div>
          </div>

        </div>

        {useCustomNN && (
          <div style={{ marginTop: 24 }}>
            {renderNNBlockBuilder()}
          </div>
        )}
      </div>
    );
  }

  // TAB 5: Bulk Predictions / Inference
  function renderPredictionsTab() {
    const filteredPredicts = predictQueue
      .filter(item => {
        if (!item || !item.path) return false;
        if (predictFilter === 'all') return true;
        if (predictFilter === 'a') return item.label_key === 'a';
        if (predictFilter === 'b') return item.label_key === 'b';
        if (predictFilter === 'unclassified') return !item.label_key;
        return true;
      })
      .sort((a, b) => {
        if (!a || !b) return 0;
        const scoreA = a.score !== null && a.score !== undefined ? a.score : -1;
        const scoreB = b.score !== null && b.score !== undefined ? b.score : -1;
        if (predictSort === 'score_desc') return scoreB - scoreA;
        if (predictSort === 'score_asc') return scoreA - scoreB;
        if (predictSort === 'name') {
          const pathA = a.path || '';
          const pathB = b.path || '';
          const nameA = pathA.split(/[\\/]/).pop() || '';
          const nameB = pathB.split(/[\\/]/).pop() || '';
          return nameA.localeCompare(nameB);
        }
        return 0;
      });

    const isAllSelected = filteredPredicts.length > 0 && filteredPredicts.every(item => item && item.path && selectedPredictPaths.includes(item.path));
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
        <div className="studio-section-header">
          <h2>Bulk Predictions</h2>
          <p>Choose folder path containing target images, calculate scores, and audit tags in bulk.</p>
        </div>

        {/* Directory selector */}
        <div className="unsloth-card" style={{ padding: '16px 20px' }}>
          <div className="flex align-center gap-16" style={{ flexWrap: 'wrap' }}>
            <div className="flex align-center gap-6" style={{ flexGrow: 1 }}>
              <Folder size={16} className="text-muted" />
              <input type="text" className="unsloth-input" style={{ height: 38 }} readOnly value={predictFolder} placeholder="Specify folders directory to run predictions..." />
              <button className="unsloth-btn unsloth-btn-secondary" style={{ height: 38 }} onClick={() => openDirectoryBrowser(predictFolder, setPredictFolder, true)}>Browse</button>
            </div>
            <button className="unsloth-btn unsloth-btn-primary" disabled={!predictFolder} onClick={loadPredictQueue}>
              Import image collection
            </button>
          </div>
        </div>

        {predictQueue.length > 0 ? (
          <div>
            {/* Inference launcher options */}
            <div className="unsloth-card" style={{ display: 'flex', alignItems: 'center', gap: 24, padding: 18 }}>
              <div className="unsloth-field-group" style={{ marginBottom: 0, width: 220 }}>
                <label>Weights classifier file</label>
                <select className="unsloth-input" value={predictModel} onChange={e => setPredictModel(e.target.value)}>
                  <option value="">-- Choose Weights --</option>
                  {workspace.models.map(m => (
                    <option key={m} value={m}>{m}</option>
                  ))}
                </select>
              </div>

              <div style={{ flexGrow: 1, display: 'flex', flexDirection: 'column', gap: 4 }}>
                {isPredicting ? (
                  <>
                    <div className="flex justify-between" style={{ fontSize: '0.75rem', fontWeight: 700 }}>
                      <span>Classifying elements...</span>
                      <span>{predictProgress}%</span>
                    </div>
                    <div className="unsloth-progress-track">
                      <div className="unsloth-progress-bar" style={{ width: `${predictProgress}%` }}></div>
                    </div>
                  </>
                ) : (
                  <span style={{ fontSize: '0.8rem', color: 'var(--muted)', fontWeight: 600 }}>
                    Load trained model weights to run bulk classifier predictions.
                  </span>
                )}
              </div>

              <div className="flex gap-6">
                {isPredicting ? (
                  <button className="unsloth-btn unsloth-btn-danger" onClick={stopInference}><Square size={12} /> Stop</button>
                ) : (
                  <button className="unsloth-btn unsloth-btn-primary" disabled={!predictModel} onClick={startInference} style={{ background: 'var(--primary)' }}><Play size={12} /> Execute inference</button>
                )}
                <button className="unsloth-btn unsloth-btn-secondary" onClick={() => setPredictQueue([])}>Clear</button>
                <button className="unsloth-btn unsloth-btn-secondary" onClick={exportPredictionCSV}><Download size={14} /> Export CSV</button>
              </div>
            </div>

            {/* Bulk Actions checkbox bar */}
            {selectedPredictPaths.length > 0 && (
              <div className="unsloth-card" style={{ padding: '12px 20px', border: '1px solid var(--primary)', backgroundColor: 'var(--primary-glow)', display: 'flex', alignItems: 'center', justifyBetween: 'space-between', marginBottom: 14 }}>
                <div style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--text-dark)' }}>
                  Selected <b>{selectedPredictPaths.length}</b> images for bulk tagging action
                </div>
                <div className="flex gap-6">
                  <button className="unsloth-btn" style={{ padding: '6px 12px', fontSize: '0.75rem', backgroundColor: 'var(--success)', color: 'white' }} onClick={() => handleBulkPredictionAction('a')}>
                    Classify all as: {activeDs.label_a}
                  </button>
                  <button className="unsloth-btn" style={{ padding: '6px 12px', fontSize: '0.75rem', backgroundColor: 'var(--danger)', color: 'white' }} onClick={() => handleBulkPredictionAction('b')}>
                    Classify all as: {activeDs.label_b}
                  </button>
                  <button className="unsloth-btn unsloth-btn-secondary" style={{ padding: '6px 12px', fontSize: '0.75rem' }} onClick={() => setSelectedPredictPaths([])}>
                    Clear selections
                  </button>
                </div>
              </div>
            )}

            {/* Filter and selector checkbox header */}
            <div className="flex justify-between align-center" style={{ marginBottom: 14 }}>
              <div className="flex gap-16 align-center" style={{ flexWrap: 'wrap' }}>
                <div className="flex align-center gap-6">
                  <span style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--muted)', display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer' }} onClick={() => {
                    if (isAllSelected) {
                      setSelectedPredictPaths([]);
                    } else {
                      setSelectedPredictPaths(filteredPredicts.map(p => p.path));
                    }
                  }}>
                    <input type="checkbox" checked={isAllSelected} onChange={() => {}} />
                    Select page
                  </span>
                </div>
                <div className="flex align-center gap-6">
                  <span style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--muted)' }}>Filter type:</span>
                  <select className="unsloth-input" style={{ padding: '4px 8px', fontSize: '0.75rem', width: 130 }} value={predictFilter} onChange={e => setPredictFilter(e.target.value)}>
                    <option value="all">Show All</option>
                    <option value="a">{activeDs.label_a}</option>
                    <option value="b">{activeDs.label_b}</option>
                  </select>
                </div>
                <div className="flex align-center gap-6">
                  <span style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--muted)' }}>Sort options:</span>
                  <select className="unsloth-input" style={{ padding: '4px 8px', fontSize: '0.75rem', width: 150 }} value={predictSort} onChange={e => setPredictSort(e.target.value)}>
                    <option value="score_desc">Confidence Desc</option>
                    <option value="score_asc">Confidence Asc</option>
                    <option value="name">Image File Name</option>
                  </select>
                </div>
                
                {/* Predictions View Toggle Capsule */}
                <div className="flex align-center gap-6">
                  <span style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--muted)' }}>Layout:</span>
                  <div className="nav-capsule" style={{ padding: 2, display: 'flex', gap: 2 }}>
                    <button 
                      className={`nav-pill ${predictViewMode === 'square' ? 'active' : ''}`} 
                      style={{ padding: '4px 10px', fontSize: '0.75rem', borderRadius: 'var(--radius-pill)', border: 'none', background: predictViewMode === 'square' ? 'var(--text-dark)' : 'none', color: predictViewMode === 'square' ? 'white' : 'var(--muted)', cursor: 'pointer' }}
                      onClick={() => setPredictViewMode('square')}
                    >
                      Square
                    </button>
                    <button 
                      className={`nav-pill ${predictViewMode === 'dynamic' ? 'active' : ''}`} 
                      style={{ padding: '4px 10px', fontSize: '0.75rem', borderRadius: 'var(--radius-pill)', border: 'none', background: predictViewMode === 'dynamic' ? 'var(--text-dark)' : 'none', color: predictViewMode === 'dynamic' ? 'white' : 'var(--muted)', cursor: 'pointer' }}
                      onClick={() => setPredictViewMode('dynamic')}
                    >
                      Dynamic
                    </button>
                    <button 
                      className={`nav-pill ${predictViewMode === 'list' ? 'active' : ''}`} 
                      style={{ padding: '4px 10px', fontSize: '0.75rem', borderRadius: 'var(--radius-pill)', border: 'none', background: predictViewMode === 'list' ? 'var(--text-dark)' : 'none', color: predictViewMode === 'list' ? 'white' : 'var(--muted)', cursor: 'pointer' }}
                      onClick={() => setPredictViewMode('list')}
                    >
                      List
                    </button>
                  </div>
                </div>
              </div>
              <div style={{ fontSize: '0.8rem', color: 'var(--muted)', fontWeight: 700 }}>
                Showing <b>{filteredPredicts.length}</b> / {predictQueue.length}
              </div>
            </div>

            {/* Predictions Display */}
            {predictViewMode === 'list' ? (
              <div style={{ backgroundColor: 'var(--panel)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', overflow: 'hidden' }}>
                <table className="model-table">
                  <thead>
                    <tr>
                      <th style={{ width: 40, textAlign: 'center' }}>
                        <input 
                          type="checkbox" 
                          checked={isAllSelected} 
                          onChange={() => {
                            if (isAllSelected) {
                              setSelectedPredictPaths([]);
                            } else {
                              setSelectedPredictPaths(filteredPredicts.map(p => p.path));
                            }
                          }} 
                        />
                      </th>
                      <th>File Name</th>
                      <th>Score</th>
                      <th>Assigned Class</th>
                      <th>Full Path</th>
                      <th style={{ width: 180, textAlign: 'right' }}>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredPredicts.map((item, idx) => {
                      const isSelected = selectedPredictPaths.includes(item.path);
                      const filename = item.path.split(/[\\/]/).pop();
                      const scorePercent = item.score !== null ? (item.score * 100).toFixed(1) + "%" : "Pending";
                      const isHigh = item.score !== null && item.score >= (workspace.settings.threshold || 50) / 100.0;
                      return (
                        <tr key={idx} style={{ cursor: 'pointer' }} onClick={() => setSelectedPredictItem(item)}>
                          <td style={{ textAlign: 'center' }} onClick={e => e.stopPropagation()}>
                            <input 
                              type="checkbox" 
                              checked={isSelected} 
                              onChange={() => togglePredictItemSelection(item.path)} 
                            />
                          </td>
                          <td style={{ fontWeight: 700, color: 'var(--text-dark)' }}>{filename}</td>
                          <td>
                            {item.score !== null ? (
                              <span className={`pred-score-tag ${isHigh ? 'high' : 'low'}`} style={{ position: 'static', border: 'none', padding: 0, background: 'none', fontWeight: 800 }}>
                                {scorePercent}
                              </span>
                            ) : (
                              <span style={{ color: 'var(--muted)', fontStyle: 'italic' }}>Pending</span>
                            )}
                          </td>
                          <td>
                            {item.label_key ? (
                              <span className={`class-label-tag ${item.label_key}`} style={{ display: 'inline-block', padding: '2px 8px', borderRadius: 4, fontSize: '0.7rem', fontWeight: 700 }}>
                                {item.label}
                              </span>
                            ) : (
                              <span style={{ color: 'var(--muted)' }}>Unclassified</span>
                            )}
                          </td>
                          <td style={{ fontSize: '0.75rem', fontFamily: 'monospace', color: 'var(--muted)', maxWidth: 400, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{item.path}</td>
                          <td style={{ textAlign: 'right' }} onClick={e => e.stopPropagation()}>
                            <div className="flex gap-4 justify-end">
                              <button className="unsloth-btn" style={{ padding: '4px 8px', fontSize: '0.65rem', backgroundColor: 'var(--success)', color: 'white' }} onClick={() => handlePredictOverride(item, 'a')}>
                                {activeDs.label_a}
                              </button>
                              <button className="unsloth-btn" style={{ padding: '4px 8px', fontSize: '0.65rem', backgroundColor: 'var(--danger)', color: 'white' }} onClick={() => handlePredictOverride(item, 'b')}>
                                {activeDs.label_b}
                              </button>
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="predictions-grid-box">
                {filteredPredicts.map((item, idx) => {
                  const isSelected = selectedPredictPaths.includes(item.path);
                  const isHigh = item.score !== null && item.score >= (workspace.settings.threshold || 50) / 100.0;
                  return (
                    <div key={idx} className={`pred-card ${predictViewMode}`}>
                      {/* Checkbox wrapper */}
                      <div 
                        className={`grid-select-checkbox ${isSelected ? 'selected' : ''}`}
                        onClick={(e) => {
                          e.stopPropagation();
                          togglePredictItemSelection(item.path);
                        }}
                      >
                        {isSelected && <Check size={12} />}
                      </div>

                      <img src={`${API_BASE}/api/fs/image?path=${encodeURIComponent(item.path)}`} alt="thumb" className="pred-image" loading="lazy" onClick={() => setSelectedPredictItem(item)} />
                      {item.score !== null && (
                        <span className={`pred-score-tag ${isHigh ? 'high' : 'low'}`} onClick={() => setSelectedPredictItem(item)}>
                          {(item.score * 100).toFixed(0)}%
                        </span>
                      )}
                      {item.label_key && (
                        <div className={`pred-label-bar ${item.label_key}`} onClick={() => setSelectedPredictItem(item)}>
                          {item.label}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}

          </div>
        ) : (
          <div className="unsloth-card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyItems: 'center', padding: '60px 20px', color: 'var(--muted)' }}>
            <Grid size={40} style={{ marginBottom: 12, opacity: 0.6 }} />
            <h4 style={{ color: 'var(--text-dark)', fontWeight: 800 }}>Predict Queue Empty</h4>
            <p style={{ fontSize: '0.85rem', marginTop: 4 }}>Select target folders and load image data to build prediction workspace.</p>
          </div>
        )}

        {/* Selected Predict detail Inspector */}
        {selectedPredictItem && renderPredictDetailModal()}
      </div>
    );
  }

  // Prediction card Zoom detail modal
  function renderPredictDetailModal() {
    const item = selectedPredictItem;
    const filename = item.path.split(/[\\/]/).pop();
    const scoreStr = item.score !== null ? (item.score * 100).toFixed(1) + "%" : "N/A";
    return (
      <div className="detail-modal">
        <div className="detail-modal-viewport">
          <img src={`${API_BASE}/api/fs/image?path=${encodeURIComponent(item.path)}`} alt="zoom-target" />
          <button className="detail-modal-close" onClick={() => setSelectedPredictItem(null)}><X size={20} /></button>
        </div>
        <div className="detail-modal-sidebar">
          <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '1.2rem', fontWeight: 800, color: 'var(--text-dark)' }}>Inference Inspector</h3>
          
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14, fontSize: '0.8rem' }}>
            <div>
              <span style={{ display: 'block', fontSize: '0.7rem', color: 'var(--muted)', fontWeight: 800, marginBottom: 4 }}>FILE NAME</span>
              <b>{filename}</b>
            </div>
            <div>
              <span style={{ display: 'block', fontSize: '0.7rem', color: 'var(--muted)', fontWeight: 800, marginBottom: 4 }}>FULL DIRECTORY PATH</span>
              <span style={{ fontFamily: 'monospace', fontSize: '0.75rem', wordBreak: 'break-all', color: 'var(--text)' }}>{item.path}</span>
            </div>
            <div>
              <span style={{ display: 'block', fontSize: '0.7rem', color: 'var(--muted)', fontWeight: 800, marginBottom: 4 }}>MODEL CONFIDENCE SCORE</span>
              <b style={{ fontSize: '1.2' + 'rem', color: item.score !== null && item.score >= 0.5 ? 'var(--success)' : 'var(--danger)' }}>
                {scoreStr}
              </b>
            </div>
            <div>
              <span style={{ display: 'block', fontSize: '0.7rem', color: 'var(--muted)', fontWeight: 800, marginBottom: 4 }}>ASSIGNED CLASSIFICATION</span>
              <span className={`class-label-tag ${item.label_key}`}>
                {item.label}
              </span>
            </div>
          </div>

          <div style={{ borderTop: '1px solid var(--border)', paddingTop: 18, marginTop: 10, display: 'flex', flexDirection: 'column', gap: 8 }}>
            <h4 style={{ fontSize: '0.85rem', fontWeight: 800, color: 'var(--text-dark)', marginBottom: 4 }}>Override classification move</h4>
            <button className="unsloth-btn" style={{ backgroundColor: 'var(--success)', color: 'white' }} onClick={() => handlePredictOverride(item, 'a')}>
              Classify as Class A: {activeDs.label_a}
            </button>
            <button className="unsloth-btn" style={{ backgroundColor: 'var(--danger)', color: 'white' }} onClick={() => handlePredictOverride(item, 'b')}>
              Classify as Class B: {activeDs.label_b}
            </button>
            <button className="unsloth-btn unsloth-btn-secondary" onClick={() => setSelectedPredictItem(null)}>
              Cancel
            </button>
          </div>
        </div>
      </div>
    );
  }

  // TAB 6: Automated Pipelines
  function renderPipelinesTab() {
    return (
      <div>
        <div className="studio-section-header">
          <h2>Automated Pipelines</h2>
          <p>Setup background processes to monitor directory folders, analyze classification ratings, and route items automatically.</p>
        </div>

        {/* Action controls bar */}
        <div className="unsloth-card" style={{ display: 'flex', alignItems: 'center', gap: 20, padding: 18 }}>
          {isBatchRunning ? (
            <button className="unsloth-btn unsloth-btn-danger" onClick={stopBatchJobs}><Square size={12} /> Terminate Pipeline Thread</button>
          ) : (
            <button className="unsloth-btn unsloth-btn-primary" style={{ background: 'var(--primary)' }} disabled={batchJobs.length === 0} onClick={runAllBatchJobs}>
              <Play size={12} /> Run Active Pipelines
            </button>
          )}

          <div style={{ flexGrow: 1, display: 'flex', flexDirection: 'column', gap: 4 }}>
            <div className="flex justify-between" style={{ fontSize: '0.75rem', fontWeight: 700 }}>
              <span>Batch processing status: <b>{batchStatusText}</b></span>
              {isBatchRunning && <span>{batchProgress}%</span>}
            </div>
            {isBatchRunning && (
              <div className="unsloth-progress-track">
                <div className="unsloth-progress-bar" style={{ width: `${batchProgress}%` }}></div>
              </div>
            )}
          </div>

          <button className="unsloth-btn unsloth-btn-secondary" onClick={addEmptyBatchJob}>
            <Plus size={14} /> Add pipeline job
          </button>
        </div>

        {/* Pipelines builder cards */}
        <div className="pipeline-list">
          {batchJobs.map((job, idx) => (
            <div key={idx} className="pipeline-card">
              <div className="pipeline-settings">
                
                {/* Row 1 input/output */}
                <div className="pipeline-settings-row">
                  <div className="unsloth-field-group" style={{ marginBottom: 0, flexGrow: 1 }}>
                    <label>Input Folder</label>
                    <div className="flex gap-6">
                      <input type="text" className="unsloth-input" style={{ height: 34, padding: '6px 10px', fontSize: '0.8rem' }} readOnly value={job.input} placeholder="Watch directories..." />
                      <button className="unsloth-btn unsloth-btn-secondary" style={{ padding: '6px 12px' }} onClick={() => openDirectoryBrowser(job.input, (path) => updateBatchJob(idx, 'input', path), true)}>Browse</button>
                    </div>
                  </div>
                  <div className="unsloth-field-group" style={{ marginBottom: 0, flexGrow: 1 }}>
                    <label>Output Folder</label>
                    <div className="flex gap-6">
                      <input type="text" className="unsloth-input" style={{ height: 34, padding: '6px 10px', fontSize: '0.8rem' }} readOnly value={job.output} placeholder="Save directories..." />
                      <button className="unsloth-btn unsloth-btn-secondary" style={{ padding: '6px 12px' }} onClick={() => openDirectoryBrowser(job.output, (path) => updateBatchJob(idx, 'output', path))}>Browse</button>
                    </div>
                  </div>
                </div>

                {/* Row 2 configs */}
                <div className="pipeline-settings-row">
                  <div className="unsloth-field-group" style={{ marginBottom: 0 }}>
                    <label>Inference Weights</label>
                    <select className="unsloth-input" style={{ height: 34, padding: '4px 8px', fontSize: '0.8rem', width: 180 }} value={job.model} onChange={e => updateBatchJob(idx, 'model', e.target.value)}>
                      <option value="">-- Choose Weights --</option>
                      {workspace.models.map(m => (
                        <option key={m} value={m}>{m}</option>
                      ))}
                    </select>
                  </div>
                  <div className="unsloth-field-group" style={{ marginBottom: 0 }}>
                    <label>Threshold (%)</label>
                    <input type="number" className="unsloth-input" style={{ height: 34, padding: '4px 8px', fontSize: '0.8rem', width: 90 }} value={job.threshold} onChange={e => updateBatchJob(idx, 'threshold', parseInt(e.target.value) || 50)} />
                  </div>
                  <div className="unsloth-field-group" style={{ marginBottom: 0 }}>
                    <label>Class A folder Name</label>
                    <input type="text" className="unsloth-input" style={{ height: 34, padding: '4px 8px', fontSize: '0.8rem', width: 110 }} value={job.label_a_name} onChange={e => updateBatchJob(idx, 'label_a_name', e.target.value)} />
                  </div>
                  <div className="unsloth-field-group" style={{ marginBottom: 0 }}>
                    <label>Class B folder Name</label>
                    <input type="text" className="unsloth-input" style={{ height: 34, padding: '4px 8px', fontSize: '0.8rem', width: 110 }} value={job.label_b_name} onChange={e => updateBatchJob(idx, 'label_b_name', e.target.value)} />
                  </div>
                </div>

              </div>

              <div>
                <button className="unsloth-btn unsloth-btn-danger" style={{ padding: 10 }} onClick={() => removeBatchJob(idx)}>
                  <Trash size={14} />
                </button>
              </div>
            </div>
          ))}

          {batchJobs.length === 0 && (
            <div className="unsloth-card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyItems: 'center', padding: '40px 20px', color: 'var(--muted)' }}>
              <FileText size={36} style={{ marginBottom: 12, opacity: 0.6 }} />
              <h4 style={{ color: 'var(--text-dark)', fontWeight: 800 }}>No Pipelines Configured</h4>
              <p style={{ fontSize: '0.85rem', marginTop: 4 }}>Click "Add pipeline job" in the control bar to construct automated task rows.</p>
            </div>
          )}
        </div>

        {/* Live console logging for pipeline execution */}
        {isBatchRunning && (
          <div className="unsloth-card" style={{ marginTop: 20 }}>
            <h3 className="unsloth-card-header" style={{ fontSize: '0.95rem', margin: 0 }}>Pipeline Run Output Stream</h3>
            <div className="console-box">
              {batchLog.map((line, idx) => (
                <div key={idx} className="console-row">{line}</div>
              ))}
              <div ref={logEndRef}></div>
            </div>
          </div>
        )}
      </div>
    );
  }

  // Directory Browser list renderer
  function renderFolderBrowser() {
    return (
      <div className="dir-modal-overlay">
        <div className="dir-modal-box">
          <div className="dir-modal-header">
            <h3>Local Folder Directory Selector</h3>
            <button className="unsloth-btn unsloth-btn-secondary" style={{ padding: 4 }} onClick={() => setShowBrowser(false)}><X size={18} /></button>
          </div>
          <div className="dir-modal-body">
            
            <div className="flex gap-6" style={{ marginBottom: 14 }}>
              <input type="text" className="unsloth-input" value={browserPath} onChange={e => setBrowserPath(e.target.value)} />
              <button className="unsloth-btn unsloth-btn-secondary" style={{ padding: '0 14px' }} onClick={() => fetchBrowserData(browserPath)}>Go</button>
            </div>

            {browserData.drives && browserData.drives.length > 0 && (
              <div style={{ marginBottom: 14 }}>
                <span style={{ display: 'block', fontSize: '0.7rem', color: 'var(--muted)', fontWeight: 800, marginBottom: 6 }}>LOGICAL DISK DRIVES</span>
                <div className="flex gap-6" style={{ flexWrap: 'wrap' }}>
                  {browserData.drives.map(d => (
                    <button key={d} className="drive-btn" onClick={() => fetchBrowserData(d)}>
                      <HardDrive size={10} style={{ marginRight: 4, display: 'inline-block', verticalAlign: 'middle' }} /> {d}
                    </button>
                  ))}
                </div>
              </div>
            )}

            <div className="dir-modal-scrollbox">
              {browserData.parent_path && browserData.parent_path !== browserData.current_path && (
                <div className="dir-modal-item" style={{ fontStyle: 'italic', color: 'var(--muted)' }} onClick={() => fetchBrowserData(browserData.parent_path)}>
                  <ArrowLeft size={14} /> .. [Go Up Parent Folder]
                </div>
              )}

              {browserData.folders.map(f => {
                const isSelected = selectedFolders.includes(f.path);
                return (
                  <div 
                    key={f.path} 
                    className="dir-modal-item" 
                    onClick={() => {
                      if (browserMultiSelect) {
                        setSelectedFolders(prev => 
                          prev.includes(f.path) ? prev.filter(p => p !== f.path) : [...prev, f.path]
                        );
                      } else {
                        setSelectedFolders([f.path]);
                        setBrowserPath(f.path);
                      }
                    }}
                    onDoubleClick={(e) => {
                      e.stopPropagation();
                      fetchBrowserData(f.path);
                    }}
                    style={{ 
                      display: 'flex', 
                      alignItems: 'center', 
                      gap: 10, 
                      padding: '6px 10px',
                      backgroundColor: isSelected ? 'var(--success-glow)' : 'transparent',
                      border: isSelected ? '1px solid var(--success)' : '1px solid transparent',
                      borderRadius: '6px',
                      cursor: 'pointer',
                      userSelect: 'none'
                    }}
                  >
                    {browserMultiSelect && (
                      <input 
                        type="checkbox" 
                        checked={isSelected} 
                        onChange={() => {}}
                        style={{ cursor: 'pointer', width: 14, height: 14 }}
                      />
                    )}
                    <div 
                      className="flex align-center gap-6" 
                      style={{ flexGrow: 1, display: 'flex', alignItems: 'center', gap: 8 }}
                    >
                      <Folder size={14} className="text-muted" style={{ flexShrink: 0 }} />
                      <span style={{ fontSize: '0.85rem' }}>{f.name}</span>
                    </div>
                  </div>
                );
              })}

              {browserData.folders.length === 0 && (
                <p style={{ color: 'var(--muted)', fontSize: '0.8rem', textAlign: 'center', marginTop: 32 }}>No subfolders located.</p>
              )}
            </div>

          </div>
          <div className="dir-modal-footer">
            <button className="unsloth-btn unsloth-btn-secondary" onClick={() => setShowBrowser(false)}>Cancel</button>
            <button 
              className="unsloth-btn unsloth-btn-primary" 
              onClick={selectBrowsedFolder} 
              style={{ background: 'var(--primary)' }}
            >
              {browserMultiSelect && selectedFolders.length > 0 
                ? `Select ${selectedFolders.length} Selected Folders` 
                : 'Select Folder'
              }
            </button>
          </div>
        </div>
      </div>
    );
  }

  // SVG Optimizer Chart drawing logic
  function renderTrainingChart() {
    if (trainStats.length === 0) {
      return (
        <div className="flex align-center justify-between w-full h-full text-muted font-bold" style={{ justifyContent: 'center', fontSize: '0.85rem' }}>
          No metrics logged. Trigger model training weights first.
        </div>
      );
    }
    
    const w = 400;
    const h = 140;
    const pad = 24;
    const cw = w - pad * 2;
    const ch = h - pad * 2;
    
    const maxE = Math.max(...trainStats.map(s => s.epoch), 1);
    const maxL = Math.max(...trainStats.map(s => s.loss), 1);
    
    const getX = (epoch) => pad + ((epoch - 1) / Math.max(maxE - 1, 1)) * cw;
    const getLY = (loss) => pad + ch - (loss / maxL) * ch;
    const getAY = (acc) => pad + ch - (acc / 100) * ch;
    
    let lossPath = "";
    let accPath = "";
    
    trainStats.forEach((s, idx) => {
      const x = getX(s.epoch);
      const ly = getLY(s.loss);
      const ay = getAY(s.accuracy);
      if (idx === 0) {
        lossPath = `M ${x} ${ly}`;
        accPath = `M ${x} ${ay}`;
      } else {
        lossPath += ` L ${x} ${ly}`;
        accPath += ` L ${x} ${ay}`;
      }
    });
    
    return (
      <svg width="100%" height="100%" viewBox={`0 0 ${w} ${h}`}>
        {/* Horizontal grid lines */}
        <line x1={pad} y1={pad} x2={w-pad} y2={pad} stroke="var(--border)" strokeWidth="1" strokeDasharray="3,3" />
        <line x1={pad} y1={pad+ch/2} x2={w-pad} y2={pad+ch/2} stroke="var(--border)" strokeWidth="1" strokeDasharray="3,3" />
        <line x1={pad} y1={pad+ch} x2={w-pad} y2={pad+ch} stroke="var(--border)" strokeWidth="1" />
        
        {/* Curves */}
        {trainStats.length > 1 && (
          <>
            <path d={lossPath} fill="none" stroke="var(--danger)" strokeWidth="2.5" strokeLinecap="round" />
            <path d={accPath} fill="none" stroke="var(--success)" strokeWidth="2.5" strokeLinecap="round" />
          </>
        )}
        
        {/* Nodes and values */}
        {trainStats.map((s, idx) => {
          const x = getX(s.epoch);
          const ly = getLY(s.loss);
          const ay = getAY(s.accuracy);
          const showLabel = trainStats.length <= 12 || idx === 0 || idx === trainStats.length - 1 || idx % Math.ceil(trainStats.length / 8) === 0;

          return (
            <g key={idx}>
              <circle cx={x} cy={ly} r="3.5" fill="var(--danger)" stroke="var(--panel)" strokeWidth="1.5" />
              <circle cx={x} cy={ay} r="3.5" fill="var(--success)" stroke="var(--panel)" strokeWidth="1.5" />
              {showLabel && (
                <>
                  <text x={x} y={ly - 6} fill="var(--danger)" fontSize="7" fontWeight="bold" textAnchor="middle" style={{ pointerEvents: 'none' }}>
                    {s.loss.toFixed(3)}
                  </text>
                  <text x={x} y={ay + 11} fill="var(--success)" fontSize="7" fontWeight="bold" textAnchor="middle" style={{ pointerEvents: 'none' }}>
                    {s.accuracy.toFixed(1)}%
                  </text>
                </>
              )}
            </g>
          );
        })}

        {/* Epoch labels */}
        {trainStats.map((s, idx) => {
          const x = getX(s.epoch);
          const showEpochLabel = trainStats.length <= 12 || idx === 0 || idx === trainStats.length - 1 || idx % Math.ceil(trainStats.length / 5) === 0;
          if (!showEpochLabel) return null;
          return (
            <text key={`ep-${idx}`} x={x} y={h - 4} fill="var(--muted)" fontSize="7" textAnchor="middle">
              Ep {s.epoch}
            </text>
          );
        })}
      </svg>
    );
  }
}
