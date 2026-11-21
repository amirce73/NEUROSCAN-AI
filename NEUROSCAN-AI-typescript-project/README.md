# 🔷 سامانه هوش مصنوعی و بینایی ماشین تشخیص تومور مغزی (بخش تایپ‌اسکریپت)
### NeuroScan AI - TypeScript Architecture & Client SDK

این پوشه شامل نسخه کامل، ماژولار و مبتنی بر تایپ‌اسکریپت (TypeScript) سامانه **NeuroScan AI** است که کلیه مدل‌ها، کنترلرها و خدمات کلاینت را با ساختار سازمانی و Type-Safety ارائه می‌دهد.

---

## 🌟 ساختار و ماژول‌های پروژه تایپ‌اسکریپت

1. **مدل‌ها و اینترفیس‌های تشخیصی (`src/types/neuroscan.ts`):**
   * تایپ‌های دقیق برای بیومارکرهای رادیومیک (`LesionBiomarkers`, `OpenCVMetrics`).
   * ساختار کارنامه انکولوژی و احتمال سرطان (`AIDiagnosticReport`, `MalignancyFactor`).
   * سیستم تایپ‌گذاری گریدهای بالینی (`WHOGrade`, `RiskLevel`, `ActiveLayer`).

2. **سرویس کلاینت ارتباطی API (`src/services/api.ts`):**
   * کلاس `NeuroScanApiClient` برای برقراری ارتباط ناهمگام با سرور پردازش تصویر، استعلام سلامت سرور، ارسال اسکن و دریافت مراحل پردازشی OpenCV.

3. **موتور رندرینگ چندلایه کنواس (`src/components/canvas-viewer.ts`):**
   * کلاس `RadiologyCanvasViewer` جهت لایه‌بندی ترکیبی (Raw CT, Brain Mask, Contours, Grad-CAM Heatmap)، کش تصاویر و مدیریت پرتو اسکن لیزری.

4. **خط‌کش کولیس دیجیتال (`src/components/caliper-ruler.ts`):**
   * کلاس `DigitalCaliperTool` برای اندازه‌گیری تعاملی درگ-اند-دراپ روی مانیتور و تبدیل آنی پیکسل به میلی‌متر بر اساس کالیبراسیون وکسل‌های اسکنر.

5. **کنترلر داشبورد و کارنامه پزشکی (`src/components/diagnostic-dashboard.ts` و `report-generator.ts`):**
   * مدیریت وضعیت واکنشی (Reactive State) برای گیج عقربه‌ای، نوارهای احتمال، بیومارکرهای کمی و چاپ کارنامه رسمی A4.

6. **نمونه کد مصرف در محیط Node.js (`src/node-demo.ts`):**
   * اسکریپت آماده جهت اتصال سایر سرویس‌های بک‌اند (NestJS, Express, Fastify) به هسته پردازش تصویر و دریافت تحلیل‌های تایپ‌شده.

7. **کنترلر اصلی برنامه (`src/main.ts`):**
   * نقطه ورود اصلی و اتصال‌دهنده کلیه کامپوننت‌ها به رویدادهای صفحه.

---

## 📁 ساختار فایل‌ها

```
typescript-project/
├── src/
│   ├── types/
│   │   └── neuroscan.ts              # تعاریف مدل‌ها، تایپ‌ها و اینترفیس‌های پزشکی
│   ├── services/
│   │   └── api.ts                    # کلاینت ارتباطی ناهمگام REST API
│   ├── components/
│   │   ├── canvas-viewer.ts          # رندرینگ چندلایه کنواس و ترکیب لایه‌ها
│   │   ├── caliper-ruler.ts          # خط‌کش کولیس دیجیتال کالیبره‌شده (mm)
│   │   ├── diagnostic-dashboard.ts   # کنترلر داشبورد ریسک و بیومارکرها
│   │   └── report-generator.ts       # کلاس تولیدکننده گزارش رسمی رادیولوژی
│   ├── node-demo.ts                  # اسکریپت دمو جهت اجرا در محیط Node.js
│   └── main.ts                       # نقطه ورود و راه‌انداز اصلی برنامه
├── tsconfig.json                     # پیکربندی کامپایلر تایپ‌اسکریپت (ES2022 Strict)
├── package.json                      # تنظیمات پکیج و اسکریپت‌های بیلد npm
└── README.md                         # راهنمای پروژه تایپ‌اسکریپت
```

---

## 🚀 نحوه اجرا و اسکریپت‌های npm

در این دایرکتوری دستورات زیر در دسترس هستند:

* **بررسی تایپ‌ها و سلامت کدها (Type Check):**
  ```powershell
  npm run type-check
  ```

* **کامپایل به جاوااسکریپت در پوشه `dist`:**
  ```powershell
  npm run build
  ```

* **راه‌اندازی سرور بک‌اند پردازش تصویر:**
  ```powershell
  npm start
  ```

