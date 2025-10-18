# Predictive Maintenance - Service Management

## 🌐 **ACCESS YOUR APP**

**Your app is always running at:** **http://localhost:8010**

Just open this link in any browser - no need to start anything!

---

## 📋 **SERVICE MANAGEMENT COMMANDS**

### Check Status
```bash
./service_status.sh
```

### Stop Service
```bash
./service_stop.sh
```

### Start/Restart Service
```bash
./service_install.sh
```

### Uninstall Service
```bash
./service_uninstall.sh
```

---

## 📝 **VIEW LOGS**

### Real-time logs
```bash
tail -f logs/stdout.log
```

### Error logs
```bash
tail -f logs/stderr.log
```

### All logs
```bash
ls -lh logs/
```

---

## 🔧 **HOW IT WORKS**

1. **Background Service**: The app runs as a macOS LaunchAgent
2. **Auto-Start**: Starts automatically when you log in
3. **Auto-Restart**: Restarts automatically if it crashes
4. **Always Running**: Works even when terminal is closed
5. **Single URL**: Always accessible at http://localhost:8010

---

## 🚀 **WHAT WAS INSTALLED**

- ✅ LaunchAgent: `~/Library/LaunchAgents/com.predictive.maintenance.plist`
- ✅ Startup Script: `start_server.sh`
- ✅ Log Directory: `logs/`
- ✅ Management Scripts: `service_*.sh`

---

## 🎯 **DEFAULT MODEL**

**Gradient Boosting** is now the default model because:
- ✅ Correctly identifies normal operations as "Working"
- ✅ 91.28% accuracy (good balance)
- ✅ Practical predictions for real-world use

**CatBoost and XGBoost** are too conservative (predict maintenance even for normal operations).

---

## 💡 **QUICK TIPS**

1. **Access anytime**: Just open http://localhost:8010 in your browser
2. **No terminal needed**: Service runs in background
3. **Check if running**: `./service_status.sh`
4. **View what's happening**: `tail -f logs/stdout.log`

---

## ⚠️ **TROUBLESHOOTING**

If the app isn't working:

```bash
# 1. Check status
./service_status.sh

# 2. Check logs
tail -20 logs/stderr.log

# 3. Restart service
./service_stop.sh
./service_install.sh

# 4. Check if port is available
lsof -i :8010
```

---

## 🎉 **YOU'RE ALL SET!**

Your Predictive Maintenance app is now running as a background service!

**Open:** http://localhost:8010

Enjoy! 🚀
