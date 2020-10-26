import smtplib
from email.mime.text import MIMEText

def email_sending(mode, dataset_name):
    if mode not in ['train', 'test']:
        raise NotImplementedError
    if dataset_name not in ['MSRVTT', 'MSVD']:
        raise NotImplementedError

    mail_host = 'smtp.163.com'
    mail_user = 'marcusnerva'
    mail_pass = 'DZGAMFOANHHSBGHF'
    sender = 'marcusnerva@163.com'
    receivers = ['hadrianus_1@163.com']

    message = MIMEText('Hello Henry. I am glad to inform you that {mode}ing on {dataset} is finished!'.format(mode=mode, dataset=dataset_name),
                       'plain', 'utf-8')
    message['Subject'] = mode + 'ing is over!'
    message['From'] = sender
    message['To'] = receivers[0]

    try:
        smtpObj = smtplib.SMTP()
        smtpObj.connect(mail_host, 25)
        smtpObj.login(mail_user, mail_pass)
        smtpObj.sendmail(sender, receivers, message.as_string())
        smtpObj.quit()
        print('email reminding is succeed!')
    except smtplib.SMTPException as e:
        print('email reminding is failed!')
        print('error', e)

def email_message(text):
    mail_host = 'smtp.163.com'
    mail_user = 'marcusnerva'
    mail_pass = 'DZGAMFOANHHSBGHF'
    sender = 'marcusnerva@163.com'
    receivers = ['hadrianus_1@163.com']

    message = MIMEText('Hello Henry. {text}'.format(text=text), 'plain', 'utf-8')
    message['Subject'] = 'NOTICE'
    message['From'] = sender
    message['To'] = receivers[0]

    try:
        smtpObj = smtplib.SMTP()
        smtpObj.connect(mail_host, 25)
        smtpObj.login(mail_user, mail_pass)
        smtpObj.sendmail(sender, receivers, message.as_string())
        smtpObj.quit()
        print('email reminding is succeed!')
    except smtplib.SMTPException as e:
        print('email reminding is failed!')
        print('error', e)