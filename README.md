<div dir="rtl">

# پنل ایدیوگرام ۴ روی مدال

پیش‌نیاز: اکانت مدال + اکانت هاگینگ‌فیس. اول لایسنس مدل را در هاگینگ‌فیس قبول کنید:

https://huggingface.co/ideogram-ai/ideogram-4-nf4-diffusers

</div>

<div dir="rtl">

گرفتن پروژه و نصب مدال (در ترمینال کیلرکودا):

</div>

```bash
git clone https://github.com/Iman8990/ideogram4-panel.git
cd ideogram4-panel
pip install modal --break-system-packages --ignore-installed
modal setup
```

<div dir="rtl">

نکته: در اوبونتوهای جدید (مثل کیلرکودا) دستور ساده‌ی pip دو ارور می‌دهد: externally-managed-environment و Cannot uninstall برای پکیج‌هایی که خود اوبونتو نصب کرده. دو فلگ بالا هر دو را حل می‌کنند و در این محیط تمرینی اشکالی ندارند.

</div>

<div dir="rtl">

ساخت سکرت‌ها (توکن هاگینگ‌فیس: Settings ← Access Tokens؛ کلید مجیک پرامپت: از ideogram.ai بخش Manage API):

</div>

```bash
modal secret create ideogram4-model-auth HF_TOKEN=xxx IDEOGRAM_API_KEY=xxx
modal secret create ideogram4-panel-auth PANEL_USER=admin PANEL_PASSWORD=xxx
```

<div dir="rtl">

کلید IDEOGRAM_API_KEY اختیاری است؛ اگر نگذارید، در پنل آپ‌سمپلر Qwen (local) را انتخاب کنید.

دیپلوی:

</div>

```bash
modal deploy gradio_ideogram4_panel.py
```

<div dir="rtl">

لینک پنل در انتهای خروجی چاپ می‌شود؛ با همان نام کاربری و رمز وارد شوید.

حذف کامل (اپ + سکرت‌ها):

</div>

```bash
modal app stop ideogram4-panel
modal secret delete ideogram4-model-auth -y
modal secret delete ideogram4-panel-auth -y
```
