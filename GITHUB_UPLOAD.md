#  GitHub Upload Guide

## ✅ What Will Be Uploaded

**Code & Documentation (small):**
- ✅ All Python source code (23 files)
- ✅ Documentation (README, guides)
- ✅ requirements.txt
- ✅ train.py
- ✅ Directory structure (.gitkeep files)

**Total size: ~30 KB** ✅

---

## 🚫 What Will NOT Be Uploaded (in .gitignore)

-  Dataset: `Celeb-DF-v2/` (9.5 GB)
-  Virtual environment: `venv/` (2.1 GB)
-  Outputs: `outputs/`, `checkpoints/`, `logs/`
-  Model weights: `*.pth`, `*.pt`
-  Cache: `__pycache__/`, `*.pyc`

**This is correct! Users will download the dataset separately.**

---

##  Step-by-Step Upload Process

### 1. Initialize Git (if not already done)

```bash
cd /home/traderx/DeepFake
git init
```

### 2. Check What Will Be Uploaded

```bash
git status
```

**You should see:**
- ✅ Source code files
- ✅ Documentation files
- ✅ requirements.txt

**You should NOT see:**
-  Celeb-DF-v2/ 
-  venv/
-  outputs/ (except .gitkeep)

### 3. Add Files to Git

```bash
git add .
```

### 4. Commit Changes

```bash
git commit -m "Initial commit: Deepfake detection system with rPPG and Hybrid CNN-Transformer"
```

### 5. Create GitHub Repository

1. Go to: https://github.com/new
2. **Repository name:** `DeepFake` (or your preferred name)
3. **Description:** "Deepfake detection using localized physiological signal inconsistency"
4. **Visibility:** Public or Private (your choice)
5. **DON'T** initialize with README (you already have one)
6. Click "Create repository"

### 6. Link to GitHub

```bash
# Replace YOUR_USERNAME with your GitHub username
git remote add origin https://github.com/YOUR_USERNAME/DeepFake.git
```

### 7. Push to GitHub

```bash
git branch -M main
git push -u origin main
```

**Done! Your code is now on GitHub!** 

---

##  If You Have Authentication Issues

### Option 1: Use Personal Access Token (Recommended)

1. Go to: https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Select scopes: `repo`
4. Generate and copy the token
5. When pushing, use token as password

### Option 2: Use SSH

```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your_email@example.com"

# Add to GitHub: https://github.com/settings/keys
cat ~/.ssh/id_ed25519.pub

# Change remote URL
git remote set-url origin git@github.com:YOUR_USERNAME/DeepFake.git
```

---

##  Verify Upload

After pushing, check:
1. Go to: `https://github.com/YOUR_USERNAME/DeepFake`
2. Verify you see:
   - ✅ Source code
   - ✅ README.md (should display nicely)
   - ✅ No dataset or venv folders

---

##  Make Your Repo Look Professional

### Add Topics/Tags

On your GitHub repo page:
1. Click "⚙️ About" (top right)
2. Add topics: `deepfake-detection`, `pytorch`, `computer-vision`, `rppg`, `transformer`, `cnn`
3. Add description: "Deepfake detection using localized physiological signal inconsistency"

### Update README on GitHub

Replace `YOUR_USERNAME` in README.md with your actual GitHub username:
```bash
# On your local machine
sed -i 's/YOUR_USERNAME/actual_username/g' README.md
git add README.md
git commit -m "Update GitHub username in README"
git push
```

---

##  Repository Stats (After Upload)

Your repo will show:
- **Language:** Python (~95%)
- **Size:** ~30 KB (code only)
- **Files:** ~25-30 files
- **Stars:** Ready for others to star! ⭐

---

##  Users Will Clone & Setup Like This:

```bash
# Clone your repo
git clone https://github.com/YOUR_USERNAME/DeepFake.git
cd DeepFake

# Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download dataset separately (instructions in README)
# Then train
python train.py
```

**Perfect! They get all the code but download dataset themselves.**

---

##  Important Notes

1. **Never upload the dataset** - It's 9.5 GB and copyrighted
2. **Never upload venv** - Users create their own
3. **Never upload model weights** - Too large, train yourself
4. **Check .gitignore** - Already configured correctly ✅

---

##  Quick Commands Reference

```bash
# Check status
git status

# See what's ignored
git status --ignored

# View remote
git remote -v

# Pull updates
git pull

# Push changes
git add .
git commit -m "Your message"
git push
```

---

**You're ready to upload! Just follow steps 1-7 above.** 
