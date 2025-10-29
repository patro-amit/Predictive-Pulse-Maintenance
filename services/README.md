# Background Service Management

This directory contains scripts and configuration files for running Predictive Pulse Maintenance as a macOS background service.

## 📁 Files

- **`service_install.sh`** - Install and start the background service
- **`service_status.sh`** - Check if the service is running
- **`service_stop.sh`** - Stop the background service
- **`service_uninstall.sh`** - Completely remove the background service
- **`com.predictive.maintenance.plist`** - macOS LaunchAgent configuration file

## 🚀 Quick Start

### Install the Service
```bash
cd /Users/shyampatro/Predictive-Pulse-Maintenance
./services/service_install.sh
```

The service will:
- ✅ Start automatically on system boot
- ✅ Restart automatically if it crashes
- ✅ Run on port 8010
- ✅ Log output to `logs/stdout.log` and `logs/stderr.log`

### Check Service Status
```bash
./services/service_status.sh
```

### Stop the Service
```bash
./services/service_stop.sh
```

### Uninstall the Service
```bash
./services/service_uninstall.sh
```

## 🔧 How It Works

The service uses macOS **LaunchAgents** to run the application in the background:

1. **`service_install.sh`** copies `com.predictive.maintenance.plist` to `~/Library/LaunchAgents/`
2. **`launchctl`** loads the configuration and starts the service
3. The service executes `start_server.sh` in the project directory
4. Logs are written to the `logs/` directory

## 📊 Service Configuration

The service is configured in `com.predictive.maintenance.plist`:

```xml
<key>Label</key>
<string>com.predictive.maintenance</string>

<key>RunAtLoad</key>
<true/>  <!-- Start on boot -->

<key>KeepAlive</key>
<true/>  <!-- Restart if crashes -->
```

## 🌐 Accessing the Application

Once the service is running:
- **Web Interface**: http://localhost:8010
- **API Endpoint**: http://localhost:8010/api/*

## 📝 Logs

Service logs are stored in:
- **Standard Output**: `logs/stdout.log`
- **Standard Error**: `logs/stderr.log`

View live logs:
```bash
tail -f logs/stdout.log
tail -f logs/stderr.log
```

## 🐛 Troubleshooting

### Service won't start
```bash
# Check for errors
./services/service_status.sh

# View error logs
cat logs/stderr.log
```

### Port 8010 already in use
```bash
# Find what's using the port
lsof -i :8010

# Kill the process
kill -9 <PID>
```

### Service starts but crashes immediately
```bash
# Check Python environment
source .venv/bin/activate
python --version

# Verify dependencies
pip list | grep -E "flask|pymongo|pyspark"

# Check logs
tail -50 logs/stderr.log
```

## 🔄 Manual Alternative

If you don't want to use the background service, you can run manually:

```bash
# Option 1: Direct execution
./run_app.sh

# Option 2: With big data processing
./bin/start_with_bigdata.sh

# Option 3: Manual Python execution
source .venv/bin/activate
python backend/app.py
```

## 📚 See Also

- [Main README](../README.md) - Project overview and setup
- [Architecture Documentation](../ARCHITECTURE.md) - System architecture
- [GitHub Setup Guide](../GITHUB_SETUP.md) - Repository configuration
