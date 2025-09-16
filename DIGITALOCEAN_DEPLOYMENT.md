# DigitalOcean Deployment Guide for CQI Backend

## Prerequisites
- DigitalOcean account
- GitHub repository with your code
- GROQ API key (for AI functionality)

## Method 1: DigitalOcean App Platform (Recommended)

### Step 1: Prepare Your Repository
1. Push all your code to GitHub
2. Ensure these files are in your root directory:
   - `requirements.txt`
   - `api_backend.py`
   - `.do/app.yaml`
   - `Procfile`

### Step 2: Create App on DigitalOcean
1. Go to [DigitalOcean Apps](https://cloud.digitalocean.com/apps)
2. Click "Create App"
3. Choose "GitHub" as source
4. Select your repository and branch (main)
5. DigitalOcean will auto-detect Python and use our configuration

### Step 3: Configure Environment Variables
In the DigitalOcean dashboard, add these environment variables:
```
GROQ_API_KEY=your_actual_groq_api_key_here
PORT=8080
```

### Step 4: Configure App Settings
- **Name**: cqi-backend
- **Plan**: Basic ($5/month recommended for this project)
- **Region**: Choose closest to your users
- **Build Command**: `pip install -r requirements.txt`
- **Run Command**: `python api_backend.py`

### Step 5: Deploy
1. Click "Create Resources"
2. Wait for build and deployment (5-10 minutes)
3. Your API will be available at: `https://your-app-name.ondigitalocean.app`

## Method 2: DigitalOcean Droplet (Manual Setup)

### Step 1: Create Droplet
1. Create Ubuntu 22.04 droplet ($6/month minimum)
2. SSH into your droplet

### Step 2: Setup Environment
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install python3 python3-pip python3-venv git nginx -y

# Clone your repository
git clone https://github.com/your-username/your-repo.git
cd your-repo

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### Step 3: Setup Environment Variables
```bash
# Create .env file
nano .env

# Add your environment variables:
GROQ_API_KEY=your_actual_groq_api_key_here
PORT=8000
```

### Step 4: Setup Systemd Service
```bash
sudo nano /etc/systemd/system/cqi-backend.service
```

Add this content:
```ini
[Unit]
Description=CQI Backend API
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/your-repo
Environment=PATH=/root/your-repo/venv/bin
ExecStart=/root/your-repo/venv/bin/python api_backend.py
Restart=always

[Install]
WantedBy=multi-user.target
```

### Step 5: Setup Nginx Reverse Proxy
```bash
sudo nano /etc/nginx/sites-available/cqi-backend
```

Add this content:
```nginx
server {
    listen 80;
    server_name your-droplet-ip-or-domain;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Step 6: Enable and Start Services
```bash
# Enable Nginx site
sudo ln -s /etc/nginx/sites-available/cqi-backend /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx

# Start your application
sudo systemctl enable cqi-backend
sudo systemctl start cqi-backend
sudo systemctl status cqi-backend
```

## Cost Comparison

### App Platform (Recommended)
- **Basic Plan**: $5/month
- **Pro Plan**: $12/month (for heavier ML workloads)
- **Pros**: Managed, auto-scaling, easy deployment
- **Cons**: Limited control

### Droplet
- **Basic**: $6/month (1 GB RAM)
- **Standard**: $12/month (2 GB RAM, recommended for ML)
- **Pros**: Full control, can optimize for ML workloads
- **Cons**: Manual setup and maintenance

## Memory Considerations

Your project needs approximately 2-4 GB RAM for full functionality:
- **App Platform**: Use $12/month plan for full ML features
- **Droplet**: Use $12/month droplet (2 GB RAM) minimum

## Testing Your Deployment

Once deployed, test these endpoints:
- `GET /api/test` - Health check
- `POST /api/analyze` - Code analysis
- `POST /api/upload` - File upload

## Troubleshooting

### Common Issues:
1. **Out of Memory**: Upgrade to higher plan or optimize requirements.txt
2. **Build Timeout**: Use Droplet method for complex builds
3. **Port Issues**: Ensure PORT environment variable is set correctly

### Logs:
- **App Platform**: Check logs in DigitalOcean dashboard
- **Droplet**: `sudo journalctl -u cqi-backend -f`

## Environment Variables Needed:
```
GROQ_API_KEY=your_groq_api_key
PORT=8080 (App Platform) or 8000 (Droplet)
```

Your CQI backend will be ready for production use!