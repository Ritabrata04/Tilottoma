sudo apt install fail2ban

# Configure SSH protection by editing /etc/fail2ban/jail.local
[ssh]
enabled = true
port = ssh
filter = sshd
logpath = /var/log/auth.log
maxretry = 3
bantime = 3600
