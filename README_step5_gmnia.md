# گام ۵: ساخت مدل GMNIA بدون اجرای تحلیل

اسکریپت `abaqus_step5_gmnia.py` فقط CAE مرحلهٔ چهارِ تولیدشده از همین pipeline مرجع را می‌پذیرد و یک فایل CAE و یک INP برای هر مدل انتخاب‌شده تحویل می‌دهد. هیچ Job ارسال نمی‌شود. مش، مختصات دارای نقص، شرایط مرزی و تماس قبلی حفظ می‌شوند. مدل بدون نقصِ مرجع نیز به‌طور پیش‌فرض وارد خروجی این مرحله می‌شود. Provenance مرحلهٔ چهار داخل Model description خوانده و قبل از هر تغییر با قرارداد `abaqus_complete_model_m20.py` کنترل می‌شود.

## دستور اجرا

کنترل مسیرهای اجرای موازی، مسیرهای معمول ویندوز، پیشوند `\\?\` و نام‌های
کوتاه 8.3 را با نرمال‌سازی و در صورت نیاز با هویت فایل‌سیستم مقایسه می‌کند.
پوشه‌های اختصاصی scratch و workspace که لینک، junction یا reparse point
باشند همچنان رد می‌شوند؛ حفاظت پیش از ساخت و پس از حل برقرار است.
`--resume` برای خطایی که پیش از ارسال Job رخ داده و هیچ نتیجه‌ای ایجاد
نکرده است، مقصد استفاده‌نشده را دوباره بررسی و اجرا می‌کند. نتایج موجود
و پوشه‌های کاری باقی‌مانده بازنویسی یا حذف نمی‌شوند.

در اجرای موازی، هر Job پوشهٔ کاری کوتاه و مستقل زیر `.queue_work` دارد.
فایل‌های scratch در مسیر کوتاهِ مستقل زیر `.abq_scratch` در پوشهٔ کد
قرار می‌گیرند؛ اگر طول مسیر کافی نباشد، مسیر TEMP بررسی می‌شود. طول مسیرهای
داخلی Abaqus پیش از ارسال Job کنترل می‌شود و هیچ تنظیمی از منابع سیستم
برای این منظور کاهش نمی‌یابد. scratch پس از توقف حلگر و تحویل خروجی پاک
می‌شود؛ در صورت توقف نامطمئن یا قفل فعال، فایل‌ها برای بررسی حفظ می‌شوند.

وجود ODB و کد خروج صفر به‌تنهایی موفقیت حل نیست. خطاهای پیش‌پردازش از
`.dat` و لاگ اجرا بررسی و با علت واقعی گزارش می‌شوند. شکست همگرایی که
منحنی قابل‌استخراج دارد، همچنان طبق قواعد فعلی نتایج ناتمام پردازش می‌شود.
ODB اولیهٔ فاقد گام تحلیل، قابل استفاده برای نمودار نیست. برای حفظ نتایج
قبلی، آن‌ها بازنویسی نمی‌شوند؛ در اجرای مقایسه می‌توان با `--resume` و
`--plates-pm-directory "Step5_PM_FY240_LG_REF1_PERFECT_FIXED"` فقط مقصد P–M
ورق‌دار را تغییر داد و مراحل قبلیِ سازگار را دوباره استفاده کرد.

### نمایش مرحلهٔ جاری در CMD

ساخت مدل و اجرای صف اکنون پیام‌های فوری همراه ساعت چاپ می‌کنند. در اجرای
`run_step5_queue_70_pct.bat`، نام و شمارهٔ مدل (`JOB 1/8`)، ارسال حل (`SUBMIT`)،
حل جاری (`SOLVING`)، خروج حلگر (`SOLVER EXIT`)، استخراج (`EXTRACT`)، رسم نمودار
(`PLOT`) و مقایسه (`COMPARISON`) مشخص می‌شوند.

ساخت Step5 خروجی را کیس‌به‌کیس تحویل می‌دهد: فایل‌های تنظیمات مشترک در ابتدا،
سپس INP بررسی‌شده و فایل `<job>_step5.json` هر کیس بلافاصله پس از تکمیل آن
در `--output-dir` قرار می‌گیرند. پیام `CASE READY i/n` تحویل هر کیس را اعلام می‌کند.
CAE مشترک فقط یک‌بار در پایان ذخیره و تحویل می‌شود؛ سپس BAT صف ساخته و تحویل
می‌شود. ساخت مدل، ذخیرهٔ CAE یا تحویل فایل‌ها هیچ تحلیلی را اجرا نمی‌کند.
اگر ساخت کیس بعدی شکست بخورد، فایل‌های تکمیل‌شدهٔ قبلی در خروجی باقی می‌مانند
و مسیر پوشهٔ موقت برای عیب‌یابی چاپ می‌شود. این تغییر قابلیت ادامهٔ خودکار ساخت
در پوشهٔ نیمه‌کامل اضافه نمی‌کند؛ اجرای مجدد همچنان پوشهٔ جدید یا خالی می‌خواهد.
فایل‌های نتایج و خروجی‌های قبلی بازنویسی نمی‌شوند.

هنگام حل، هر ۱۵ ثانیه زمان سپری‌شده، آخرین خط فایل `.sta` (در صورت وجود)،
و در حالت monitor آخرین LPF و نسبت بار به بیشترین بار ثبت‌شده نمایش داده می‌شود.
این نسبت درصد پیشرفت تحلیل نیست. تا پیش از ایجاد ODB پیام `Waiting for ODB`
نمایش داده می‌شود. خروجی کامل حلگر همچنان در `<job>_queue_solver.log` ذخیره می‌شود.
پیام‌های کنسول انگلیسی‌اند تا در CMD به‌هم نریزند.

برای تغییر فاصلهٔ نمایش، `--progress-seconds 5` را به انتهای خط فراخوانی
`call abaqus ...` در BAT اضافه کنید؛ این گزینه فاصلهٔ پایش شرط توقف را تغییر نمی‌دهد.

برای سازگاری با نصب‌هایی از Abaqus/CAE که stdout/stderr اسکریپت noGUI را تا پایان process
در CMD عبور نمی‌دهند، BAT جدید گزارش‌ها را هم‌زمان در
`STEP5_queue_progress.log` می‌نویسد و `step5_progress_tail.ps1` همان فایل را در همان
پنجرهٔ CMD به‌صورت زنده نمایش می‌دهد. این مسیر فقط presentation/monitoring است و هیچ
تغییری در solver، ODB، ترتیب صف یا معیار توقف 70% ایجاد نمی‌کند.
تنظیمات تحلیل و معیار توقف همان مقادیر ورودی‌اند. کد به‌روزشده در اجرای بعدی
بارگذاری می‌شود؛ اجرای از قبل در حال کار پیام‌های جدید را دریافت نمی‌کند.

در مثال زیر **350 صرفاً نمونهٔ ورودی fy** برحسب MPa است؛ مقدار مناسب مصالح خودتان را وارد کنید. پوشهٔ خروجی باید جدید یا خالی باشد.

```bat
cd /d "D:\CFS-Column\New folder (6)"
abaqus cae noGUI=abaqus_step5_gmnia.py -- ^
  --source-cae "D:\CFS-Column\New folder (7)\Step4_imperfections.cae" ^
  --output-dir "D:\CFS-Column\New folder (7)\Step5_FY350" ^
  --fy 350
```

با این دستور هر هفت مدل `STEP4_*` و یک مدل بدون نقص ساخته می‌شوند. برای شروع با یک مدل، به دستور اضافه کنید:

```bat
  --models STEP4_L_low
```

اگر دستور چندخطی است، `^` فقط در انتهای هر خطِ غیرآخر باشد. برای چند مدل، نام‌ها را پس از `--models` با فاصله بنویسید. برای fy دیگر، دستور را با fy و پوشهٔ خروجی جدید اجرا کنید.

## محتوای مدل

- Step از نوع `Static, Riks` با `NLGEOM=ON`؛ Buckle قبلی حذف می‌شود.
- مصالح به‌طور پیش‌فرض الاستیک–کاملاً پلاستیک با fy ورودی؛ با `--stress-strain` منحنی سخت‌شوندگی (پایین را ببینید). E و ضریب پواسون قبلی حفظ می‌شوند. شکست یا آسیب مصالح و پیچ‌ها تعریف نشده است.
- فشار مرجع پیش‌فرض برابر fy است. `--reference-stress` امکان تغییر آن را می‌دهد.
- مدل مرجع Step 3 از **BEAM_MPC** استفاده می‌کند. Step 4 این اتصالات را بدون تغییر نگه می‌دارد. در Step 5، فقط برای دسترسی به نیروی/گشتاور اتصال، هر BEAM_MPC به **Assembled BEAM Connector** روی دقیقاً همان دو گره تبدیل می‌شود. تعداد لینک‌ها و endpointها با provenance مدل مرجع کنترل می‌شوند و هیچ compliance عمدی اضافه نمی‌شود. این تبدیل در metadata به‌صورت صریح ثبت می‌شود و نباید به‌عنوان Fastener جدید یا تغییر footprint تلقی شود.
- محور محلی ۱ هر اتصال از گره A به B، محور ۲ در راستای تصویر محور طول ستون است. `CTF1..3` نیروها و `CTM1..3` گشتاورها در این دستگاه هستند.
- تماس Hard/Frictionless موجود، تکیه‌گاه‌های دو انتها و مهار محوری میانی حفظ می‌شوند.
- خروجی‌های `U`, `RF`, `S`, `LE`, `PEEQ`، فشار/بازشدگی تماس، و نیروی اتصال‌ها تعریف شده‌اند. تنش و کرنش پلاستیک در همهٔ نقاط ضخامت پوسته درخواست می‌شوند. LPF توسط Riks خودکار نوشته می‌شود.
- هندسهٔ اصلی قطعه همچنان بدون نقص است؛ نقص در مش قرار دارد. **دوباره مش‌بندی نکنید.**

## تنظیمات شروع

| گزینه | پیش‌فرض |
|---|---:|
| `--initial-arc` | 0.01 |
| `--min-arc` | 1e-8 |
| `--max-arc` | 0.05 |
| `--max-increments` | 5000 |
| `--max-end-displacement-mm` | غیرفعال (فقط با مقدار صریح فعال می‌شود) |
| `--field-frequency` | 1؛ هر Increment |
| `--cpus` | `auto` = همهٔ هسته‌های فیزیکی؛ یا یک عدد مثبت، مثلاً `--cpus 4` |

`--cpus` در Jobها، در `abaqus_v6.env` و در `run_step5_queue_*.bat` تولیدشده نوشته می‌شود؛ صف آن را روی
همان رایانه‌ای که حل را اجرا می‌کند دوباره تعیین می‌کند (درخواست بیش از تعداد هسته‌های فیزیکی، همهٔ آن‌ها را
به‌کار می‌گیرد و هشدار می‌دهد). برای اجرای صف با تعداد دیگری، `--cpus N` را در خط `call abaqus ...` همان BAT
عوض کنید. همهٔ گزینه‌ها به شکل `--name value` (با فاصله) نوشته می‌شوند؛ لانچر Abaqus شکل `--name=value` را
پیش از شروع اسکریپت رد می‌کند. در یک دستور چندخطی با `^` خط خالی نگذارید.

شرط جابه‌جایی مربوط به **U3 مثبت یک گره در انتهای پایین** است؛ کوتاه‌شدگی کل ستون نیست. گره پایش در Set به نام `STEP5_STOP` مشخص است. اندازهٔ مش از مرحلهٔ چهار می‌آید و اینجا تغییر نمی‌کند. طول‌قوس، زمان فیزیکی تحلیل نیست. معیارهای تعادل پیش‌فرض Abaqus ضعیف نشده‌اند و پایدارسازی مصنوعی اضافه نشده است.

این‌ها تنظیمات اولیه‌اند؛ ساخت موفق فایل‌ها، تضمین همگرایی یا صحت پژوهشی تحلیل نیست. پس از حل باید ثبت بار اوج و بخش کافی از شاخهٔ نزولی، تعادل، رفتار تماس و حساسیت عددی بررسی شود. پایان به‌دلیل سقف Increment را خودکار «ظرفیت نهایی به‌دست آمد» تلقی نکنید.

## استخراج نتایج در مرحلهٔ بعد

در توضیح هر Model، تنظیمات و `reference_force_N_per_end` ثبت می‌شوند:

`P = LPF * reference_force_N_per_end`

این نیرو مربوط به **یک انتها** است. نیروهای دو انتها را با هم جمع نکنید و واکنش مهار میانی را بار ستون نگیرید. سطح مرجع از لبه‌های مش S4R محاسبه می‌شود و ممکن است اندکی با مساحت هندسی اسمی تفاوت داشته باشد.

کوتاه‌شدگی سازگار با بارگذاری توزیع‌شده:

`delta = area_weighted_mean(U3_bottom) - area_weighted_mean(U3_top)`

وزن‌ها در `end_area_weights` توضیح Model ذخیره می‌شوند. History خروجی جابه‌جایی دو انتها و نیرو/گشتاور تمام پیچ‌ها در هر Increment موجود است؛ بنابراین نیروی پیچ‌ها در Increment بار اوج قابل بازیابی خواهد بود. کد حاضر فقط مدل می‌سازد و نمودار یا ODB تولید نمی‌کند.

فایل CAE نهایی همهٔ مدل‌ها و Jobهای آماده را دارد؛ هر INP به یک Job تعلق دارد.
پوشهٔ خروجی INP و metadata هر کیس را به‌تدریج و CAE و BAT را در پایان دریافت می‌کند؛
تنظیمات مشترک و فایل URDFIL نیز در آن قرار دارند. فایل‌های شروع خود Abaqus مانند
`abaqus.rpy` ممکن است در پوشهٔ اجرای دستور ایجاد شوند.


## قرارداد سازگاری با pipeline مرجع

نسخهٔ فعلی Step 4/5 به فایل‌ها بر اساس wildcard یا ترتیب پوشه اعتماد نمی‌کند. Step 4 فایل
`pipeline_status.json` را از `--run-dir` می‌خواند و فقط CAE/ODB مربوط به
`build.job_name` همان run را می‌پذیرد. پیش از اعمال imperfection موارد زیر کنترل می‌شوند:

- مدل و Job مرجع ثبت‌شده توسط `abaqus_complete_model_m20.py`؛
- چهار instance دقیق `P1..P4`؛
- المان `S4R`؛
- تعداد BC و load مطابق build مرجع؛
- General Contact با Hard/Frictionless؛
- اتصال `BEAM_MPC` و تعداد لینک‌ها مطابق `expected_links`؛
- CAE و ODB متعلق به همان basename مرجع.

Step 4 این قرارداد، تنظیمات mesh/longitudinal lines و SHA-256 فایل‌های مرجع را در
Model description مدل‌های `STEP4_*` ذخیره می‌کند. Step 5 بدون این provenance اجرا
نمی‌شود؛ بنابراین CAE قدیمی که با نسخهٔ مستقل قبلی Step 4 ساخته شده است باید با
نسخهٔ جدید Step 4 دوباره ساخته شود.

Jobهای Step 5 نیز precision خروجی nodal را از run مرجع به ارث می‌برند و همان
تنظیم حافظهٔ ۱۰۰٪ بدون رزرو و همهٔ CPUهای در دسترس را استفاده می‌کنند. CAE و INP هر مدل از یک Model/Job
واحد تولید می‌شوند؛ نسخهٔ نمایشی جدا از مدل تحلیل وجود ندارد.


## Force-displacement extraction

For a solved STEP5 ODB, use `abaqus_step5_force_displacement.py` rather than
plotting a single nodal RF3. The STEP5 end nodes are not axially constrained,
and the reference load is applied as shell edge traction, so an end-node RF3 is
not the column load.

The extractor uses exactly the conventions stored by the STEP5 builder:

`P = LPF * reference_force_N_per_end`

and

`delta = area_weighted_mean(U3_bottom) - area_weighted_mean(U3_top)`.

It reads the reference force and end-area weights from the matching STEP5 CAE,
then reads the automatic Static-Riks `LPF` history output and U3 from the solved
ODB for every GMNIA frame. The ODB frame value is used only to align the frame
with its LPF history sample; it is not assumed to equal LPF. It writes CSV,
JSON, PNG and a dependency-free SVG curve, and reports the maximum recorded load.

Example:

```bat
cd /d "C:\Users\810200014.HAMI.000\Documents\Abaqus_run"

abaqus cae noGUI=abaqus_step5_force_displacement.py -- ^
  --cae "D:\CFS-Column\New folder (7)\New folder\Step5_GMNIA_FY240\Step5_GMNIA_FY240.cae" ^
  --odb "D:\CFS-Column\New folder (7)\New folder\Step5_GMNIA_FY240\STEP5_D_FY240.odb"
```

The default output directory is the ODB directory. Expected files are:

- `STEP5_D_FY240_force_displacement.csv`
- `STEP5_D_FY240_force_displacement.json`
- `STEP5_D_FY240_force_displacement.svg`
- `STEP5_D_FY240_force_displacement.png`

The SVG is publication-friendly vector output. The CSV can be imported directly
into Excel, Origin, MATLAB or Python for final figure styling.

For an analysis still running, append `--allow-partial`. The ODB is opened
read-only even when its `.lck` exists; no analysis is submitted or terminated.
Only the synchronized prefix of LPF history and displacement frames is plotted.
Live snapshots use `_force_displacement_partial.csv/.json/.svg/.png` names and
the chart is labelled `PARTIAL SNAPSHOT`. The recorded peak may change as the
analysis continues. Run the command again to refresh the snapshot. If the ODB
or its first LPF/U outputs have not yet been flushed, retry after more increments.
After completion, rerun without `--allow-partial` to produce final filenames.

`abaqus_step5_force_displacement.py` is the noGUI entry script and invokes
the extractor explicitly even in Abaqus execution namespaces other than
`__main__`. Python callers should import `abaqus_step5_force_displacement_core`.
PNG uses Matplotlib when available, with native CAE plotting as fallback.


## Riks robustness and Dynamic Explicit

Completion is checked independently of the launcher exit code and post-peak
criterion. The queue combines this job's `.sta`, `.msg`, `.log`, captured solver
stdout and available ODB diagnostic status. A `.sta` without a terminal footer
does not prevent checking the other sources. Status is refreshed after locks are
released and extraction closes the ODB. Explicit failure or incomplete-ending
evidence overrides success; a recorded native termination with a verified crossing
is handled separately. A terminate message for PERFECT never establishes the
status of another imperfection job. `analysis_state`, `completion_sources` and
`completion_diagnostics` identify the evidence in queue JSON.

Resume re-evaluates legacy `INCOMPLETE_ANALYSIS`/`SOLVER_STATUS_UNKNOWN` labels
against current terminal evidence and the saved numerical curve. Cached ODB job
status is used only when its recorded ODB size/timestamp still match. Missing
completion proof stays UNKNOWN; zero exit code, synchronized frames or a plausible
curve alone do not establish successful execution. Reassessment does not fabricate
a verified capacity, overwrite solver artifacts or submit completed jobs again.

Static Riks remains the default and preserves the existing force-controlled end
loading, imperfections, warping freedom, contacts, materials and LPF postprocessing.
For an optional change to the nonlinear initial predictor, append to the Step5
builder command (or the family pipeline command):

```bat
  --riks-extrapolation none ^
```

The default `linear` leaves Abaqus's original predictor unchanged. `none` suppresses
extrapolation; it can help when the previous incremental direction is a poor guess
after a sudden stiffness/contact change, but is not a guaranteed convergence fix.
It adds no inertia, damping or compliance and does not relax equilibrium tolerances.
The choice is recorded in Step5 metadata. Pipeline resume accepts legacy metadata
as `linear` and refuses a different requested predictor for an existing build;
changing it requires a new output directory, not altering completed analyses.

`totalArcLength=1` is the Riks arc-length **scale factor**, not a stopping condition.
Normal termination can use an LPF/displacement limit or the maximum increment count;
the queue's observed post-peak crossing is a separate termination request. An unmet
post-peak criterion alone therefore does not identify a convergence failure or its
cause. Read the failed job's `.msg`/`.sta` diagnostics before changing the solver.

A drop-in Dynamic Explicit switch is intentionally not implemented. This builder
has no dynamic density/time definition, inherits Standard general contact, and
uses dead end loads with LPF-based force/moment extraction and stop monitoring.
Replacing only the step would not provide a valid quasi-static capacity workflow.
In particular, a monotonically prescribed force/moment amplitude does not measure
the descending resistance curve. Prescribing rigid end-section kinematics instead
would alter the current warping freedom and cannot be substituted silently.

A defensible Explicit alternative would require a compatible contact/connector
formulation, consistent physical density, smooth displacement/generalized-rotation
control that preserves the intended end freedoms, reaction-based resistance and
stopping, and energy/rate verification. Inspect kinetic/internal, artificial and
dissipative energies near the peak as well as force equilibrium; a low global
kinetic/internal-energy ratio alone does not establish capacity independence from
loading rate or mass scaling. No automatic mass scaling or damping is introduced.
Such a conversion and its rate-sensitivity checks exceed this minimal solver-control
change; no Explicit ultimate-capacity result is claimed.

References: [Abaqus Riks settings](https://docs.software.vt.edu/abaqusv2023/English/SIMACAECAERefMap/simacae-t-simconfigureriks.htm)
and [Abaqus buckling with general contact](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEBMKRefMap/simabmk-c-sscxsec.htm).

## Perfect reference and 70% post-peak stop

The Step-5 builder now includes the untouched no-imperfection reference model by
default, in addition to all `STEP4_*` imperfection cases. For `fy=240` the
perfect model is named:

```text
STEP5_PERFECT_FY240
```

The perfect model is copied from the untouched reference model stored in the
Step-4 CAE. No nodal imperfection is added; the same mesh, S4R formulation,
contact, boundary conditions and original BEAM_MPC bolt endpoints are preserved
before the normal Step-5 conversion to assembled BEAM connectors.

All Step-5 jobs also use one common post-peak stopping rule. The default is:

```text
P <= 0.70 Pu
```

at the first converged increment after the peak. Since the Step-5 loading has no
preload and uses `P = LPF * P_ref`, this is exactly equivalent to:

```text
LPF <= 0.70 * LPF_peak
```

The default queue now uses **compiler-free Riks `.sta` monitoring**. Abaqus
writes the converged total LPF to the append-only status file while Standard is
running, so the queue does not need to open a solver-owned live ODB. Unsuccessful
attempt lines such as `1U` are ignored. Because the printed `.sta` LPF is rounded,
the live stop uses a conservative rounding bound: termination is requested only
when the upper bound of the current printed LPF is at or below 70% of a lower
bound on the previously observed peak. Thus text rounding cannot terminate the
analysis before the true 70% crossing. The final ODB extraction still uses the
automatic full-precision Riks LPF history and independently verifies the crossing.

The queue invokes native `abaqus terminate job=...` after that observed crossing.
No `user=` is passed, so `ifort` is not needed. This is external monitoring,
not an increment callback: status-file polling and command latency can let extra
increments run. Do not claim exact first-increment termination for monitor mode.
Native termination cannot be resumed as a suspended job.

`--stop-method urdfil` retains the exact increment callback option and requires
a compatible Intel Fortran + Visual Studio Abaqus user-subroutine environment.
The routine reads actual LPF from `.fil` record 2000 attribute 9 rather than
assuming the URDFIL step-time argument is LPF. The builder still supplies the
Fortran file and minimal frequency=1 NODE FILE request for this option.

The default maximum is 5000 increments, with no displacement stop unless
`--max-end-displacement-mm` is explicitly given. Nonconvergence and other solver
failures remain failures; they are not converted into successful completion.

## Repair a verified queue before any solver artifacts exist

New sidecars bind each job name to the SHA256 of the verified final INP. A
modified input or an old sidecar without this binding cannot be submitted by
the queue: rebuild in a new directory. Legacy `--extract-only` remains available
for reading prior outputs. URDFIL additionally requires the builder's stop ratio
and, when recorded, its exact subroutine hash; monitor mode may use another
ratio. Batch commands retain full floating-point precision. Existing solver
artifacts block a new solve and are preserved without backups. Duplicate initial
zero LPF samples are normalized before a live common-prefix extraction. Near-zero
P/M endpoint resultants are clamped only within section-scaled roundoff tolerance.

In a newly generated directory containing verified `STEP5_*.inp` sidecars, run:

```bat
abaqus cae noGUI="D:\Abaqus_run\abaqus_step5_queue_nogui.py" -- --run-dir . --repair-only --cpus 4
run_step5_queue_70_pct.bat
```

Repair validates the verified job metadata and rewrites the
queue (same file name as the builder, from `--ratio`; `--cpus`/`--gpus` are written into it); it does not rebuild/modify CAE or INP and does not submit jobs. The new BAT
references the repository runner by absolute path. Keep that repository path
available. Existing ODBs and live locks are never overwritten/deleted by this
runner. Existing solver artifacts are preserved in place and submission is refused when a destination is occupied; use a new run directory/job name. No backup copies are created. For completed/failed existing ODBs, preserve results and replot using:

```bat
abaqus cae noGUI="C:\Users\810200014.HAMI.000\Documents\Abaqus_run\abaqus_step5_queue_nogui.py" -- --run-dir . --extract-only
```

If several STEP5 CAEs are present, specify the exact `--cae` path. Use `--jobs`
to restrict the exact jobs intentionally. A failed earlier job does not prevent
attempting the remaining queue, except that preflight conflicts stop submission
before any job starts. Solver stdout/stderr is saved per job as
`STEP5_*_queue_solver.log`.

## Bounded Newton controls for new inputs

New Step5 inputs use `--riks-controls robust`: Newton line search with at most
four searches and `ANALYSIS=DISCONTINUOUS` (I0=8, IR=10). The default maximum
16 equilibrium iterations, five increment attempts, minimum arc 1e-8 and
5000-increment ceiling remain unchanged. No field/constraint accuracy tolerances,
hardening, damping, supports, loads or geometric imperfections are changed.
The line searches add element residual evaluations, not stiffness decompositions.
This is a bounded correction strategy, not an unlimited solver retry sequence.

Family/comparison/HSS pipelines default to `--riks-controls auto`: robust for
new inputs, recorded controls for occupied inputs. Legacy metadata without this
key means default Abaqus controls. `--riks-controls default` explicitly retains
those original controls. Saved failed results remain failed and are not silently
resubmitted or overwritten when source code changes.

For a new numerical attempt with verified earlier stages, use `--resume` with
`--riks-controls robust --pm-directory Step5_PM_ROBUST_V1` (choose an unused name).
The common comparison directory option also reaches HSS. Standalone Step5 uses
a fresh `--step5-directory` instead. Earlier Step4 provenance can be reused even
when its old nonlinear delivery used default controls. Occupied destinations
still reject a different explicitly requested policy.

These controls cannot guarantee postbuckling continuation through a true
perfect-model bifurcation. A converged observed response, rather than a solver
exit code or merely decreasing LPF, is required for interpreting a selected point.
Ordinary unfinished-curve selection is unchanged; no missing P-M point is invented.
See the official [Newton convergence controls](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEANLRefMap/simaanl-c-convergcriteria.htm)
and [Riks bifurcation restrictions](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEANLRefMap/simaanl-c-postbuckling.htm).

## Automatic plots and resource requests

The queue extracts each finished/terminated readable ODB with its own matching
CAE model metadata. It writes per-job CSV, JSON, SVG and PNG. It updates these
comparison outputs after each attempted job and at the end:

- `STEP5_all_force_displacement_combined.png` (Matplotlib 300 dpi when available)
- `STEP5_all_force_displacement_combined.svg`
- `STEP5_peak_summary.csv`
- `STEP5_queue_status.json`

Plots use shortening in mm, compression in kN, a legend including PERFECT and
an `AsFy` reference computed from the model end area and its actual fy. Cases
that do not reach the 70% drop or have solver errors are labeled accordingly.
No strain/displacement sorting is applied to Riks snap-back paths. All cases
must have field_frequency=1 for audited sequence alignment with automatic LPF
history. The builder enforces this; existing sparse-output models are rejected.

An apparent 70% LPF crossing is also screened against the preceding ascending
response. If every available active end-response channel retraces that path
within 1% of its pre-peak excursion, the peak is marked `PATH_REVIEW_REQUIRED`:
the observed increment is retained for P-M display as `UNVERIFIED_PATH_RETRACE`,
but is not certified as Pu/Mu. A decreasing displacement alone does not trigger
this warning; real snap-back need not retrace the ascending path. This screen
does not establish physical collapse or rule out bifurcation. The established
selection policy for other unfinished curves is unchanged. Resume reassesses
saved CSVs and refreshes reports without submitting an existing occupied job.

If Matplotlib is unavailable in Abaqus Python, PNG uses the native CAE XY plotting
engine; SVG always uses the standard library. No pip installation is required.
Native Abaqus/Windows execution and PNG fallback require verification on your
machine; unit tests here cover numerical logic, resource commands and mocked
process boundaries, not an actual GMNIA solve.

New reference buckling and Step5 jobs default to all physical CPU cores, memory=100%
with getMemoryFromAnalysis disabled, and all detected NVIDIA GPUs requested
through documented Abaqus CLI/environment settings (not an unsupported Job
constructor keyword). Step5 requests standard_parallel=all. Buckling retains
standard_parallel=solver because that procedure does not support parallel
Standard element operations. Explicit --cpus/--gpus overrides (e.g. --cpus 4) are
accepted by Step 5, the queue runner, the P-M modes and the Step-1..3 builder, and
are carried into every generated BAT; --cpus auto uses all physical cores.
No fixed RAM reserve or worker cap is imposed. The solver only uses hardware
supported by its procedure and installed drivers; 100% utilization of every
resource throughout a solve is not guaranteed.

The BAT now launches `abaqus_step5_queue_nogui.py`, an explicit CAE entry script
that calls the reusable runner main even when CAE does not set `__name__` to
`__main__`. Existing output BATs are generated artifacts and are not modified by
`git pull`: run the repair-only command once after this launcher update. A fresh
run must print `STEP5 NOGUI DRIVER STARTED`, then its resource request and job
progress; exit code zero alone is not evidence that the queue executed.

Abaqus on the target Windows machine reports 12 available solver CPUs despite
24 logical SMT processors. Solver auto mode therefore uses all physical cores
(12 on this machine); Python mode screening still uses all logical processors
(24). No core is reserved. Queue CLI CPU settings override older generated env
files, so existing CAE/INP/BAT files need not be rebuilt for this change. A launcher
exit code zero with no ODB now reports the actual solver log as SOLVER_ERROR.


## Unified family entry point

For compression, bending, selected PERFECT/LG_mp cases, report references and
an integrated nonlinear P-M queue, see the
[unified no-added-plate commands](README.md#unified-family-commands-for-the-model-without-added-plates).
The standalone commands in this document remain supported. Step 5 also accepts
`--cufsm-report-dir DIR`; it stores the built-up report references in each job's
metadata. This option adds graph references without changing loads or geometry.
The family CLI's `--model-cases perfect|imperfect|both` forwards the existing
standalone `--models PERFECT`, `--models STEP4_LG_mp`, or both, respectively.
CAE readers use retained working files so the source CAE is not modified merely
by opening it in the kernel.

## Bending (moment-rotation)

A Step-4 CAE built from a bending run is detected automatically from its provenance; the command is the
same as for compression. Differences:

- Loading: the Step-3 consistent nodal end forces of the linear stress `1 MPa*eta/c` (`BEND_END_*`, CF3)
  are recomputed from the end nodes, checked against the inherited loads (names, nodes, count, applied
  moment, zero net axial force) and scaled to `--reference-stress` (default fy). LPF = 1 is therefore the
  first-yield moment `M_y` at the extreme centre-line fibre, and `M = LPF*reference_moment_Nmm` (no preload).
- Response: work-conjugate end rotation `theta = sum_bottom k_i U3_i + sum_top k_i U3_i`,
  `k_i = F_i/M_ref` (signed, stored as `end_rotation_weights`). It equals the relative rotation of plane end
  sections exactly, a uniform U3 does not contribute, and `M*theta` is the work of the end forces
  (positive = compression fibre shortens).
- Post-peak stop: the same LPF ratio, i.e. the first converged increment after the peak with `M <= 0.70 Mu`.
- Outputs: `<job>_moment_rotation.csv/.json/.svg/.png`, `STEP5_all_moment_rotation_combined.*`, and
  `STEP5_peak_summary.csv` with `moment_kNm`, `My_kNm`, `Mu_over_My`, `Mp_kNm`, `Mu_over_Mp`.
  `M_p = fy*Z` with the centre-line thin-walled plastic modulus.
- P-M: every Step-5 folder (compression and bending) writes `STEP5_PM_points.csv`
  (`load_case, bending_axis_deg, P_kN, M_kNm, Py_kN, My_kNm, Mp_kNm`, amplitudes, outcome): one capacity
  point per model, ready to be combined across the compression and the 0/45 deg bending folders.

```bat
set "RUN_DIR=D:\CFS-Column\<input>_BEND000"
cd /d "C:\Users\810200014.HAMI.000\Documents\Abaqus_run"
abaqus cae noGUI=abaqus_step5_gmnia.py -- ^
  --source-cae "%RUN_DIR%\imperfection\Step4_LDG.cae" ^
  --output-dir "%RUN_DIR%\imperfection\Step5_GMNIA_FY240_LDG" ^
  --fy 240 ^
  --initial-arc 0.01 ^
  --min-arc 1e-8 ^
  --max-arc 0.05 ^
  --max-increments 5000 ^
  --field-frequency 1 ^
  --postpeak-stop-ratio 0.70
```

## Metadata sidecars

Step 5 inherits the loading from the Step-4 provenance. Optional `--load-case bending`
(or `compression`) asserts that recorded case and stops on a mismatch; it never converts
one load case into the other. Bending uses opposite consistent nodal end-force distributions,
checks the primary moment, zero net axial force and absence of a moment about the other axis,
then scales those forces to `--reference-stress` (default `--fy`). The two residual ratios
are recorded as `net_axial_force_relative` and `secondary_moment_relative` in the sidecar.

Every Step-5 job now also gets `<job>_step5.json` next to its INP: the same JSON that is stored in the
model description of the CAE. The queue and `abaqus_step5_force_displacement.py` read the sidecar first,
so `--cae` is only needed for older folders without sidecars. P-M interaction jobs
(`abaqus_pm_interaction.py --mode gmnia`, see README.md) use the same queue and have sidecars only.

## Strain hardening (optional `--stress-strain`)

By default the material is elastic-perfectly-plastic at `--fy`. If coupon data are available, give the
engineering stress-strain curve as a CSV file with two columns, engineering strain and engineering stress in
MPa. A header row is allowed.

```text
strain,stress_MPa
0.0012,240
0.02,290
0.10,340
```

- The first row is the yield point: its stress must equal `--fy` (within 0.5 %), and its strain must not
  exceed the elastic strain fy/E.
- Step 5 converts every row to true stress = s(1+e) and logarithmic plastic strain = ln(1+e) − true stress/E,
  which is the form `*PLASTIC` requires with NLGEOM.
- Plastic strain must increase and true stress must not decrease (softening would need a damage model).
- Beyond the last row the stress stays constant.
- The table and the CSV checksum are recorded in each model description (`plastic_tables`, `material`).

```bat
  --stress-strain "D:\CFS-Column\coupon_FY240.csv" ^
```

Corner strength enhancement from cold work and residual stresses are not modelled. Most modal-imperfection
GMNIA studies of cold-formed steel follow the same convention: the residual stress effect is treated as part
of the equivalent geometric imperfection.

## Peak summary for imperfection-sensitivity studies

`STEP5_peak_summary.csv` (written by the queue) now also lists, for every model:
- `a_L_mm`, `a_D_mm`, `a_G_mm`: the Step-4 amplitudes of u0 = aL·phiL + aD·phiD + aG·phiG;
- `Py_kN` = A·fy;
- `Pu_over_Py`.

With the `--combo` cases of Step 4, this gives the strength-versus-amplitude table directly.

## How to read the PERFECT model

`STEP5_PERFECT_FY…` has no imperfection. Riks then follows the symmetric fundamental path until yielding, or
until round-off triggers a bifurcation. Its peak is an upper reference, used for normalisation and for
checking the imperfection sensitivity; it is not a design strength.

## Shared P-M selection after a stopped analysis

Response-v3 uses the same selection for independent no-plate jobs and the
plated/HSS comparison. Existing significant load-drop and stiffness-change
events are retained, followed by the verified peak or incomplete ascending
endpoint as applicable. Otherwise a terminal response contributes its observed
maximum from one synchronized converged P/M frame, labelled
`UNVERIFIED_OBSERVED_MAXIMUM` with `capacity_verified=false`.
All displayed points participate in their own specimen/case connection,
including provisional points. Those lines are comparison guides, not verified
resistance envelopes. Failed analyses retain their solver-error status.
Running jobs or missing/invalid curves cannot supply a completed point.
Collection/resume can reassess saved CSVs without submitting an occupied job.

## Per-P-M-job response pairs and ODB frames

P-M collection writes `diagram/response_curves/<dataset>_<identity>/<job>_axial_bending_response.png`
and `.svg`: P (kN) versus axial shortening (mm) on the left and M (kN m)
versus total end rotation (mrad) on the right. Every requested perfect/imperfect
job with a validated saved history is included, including stopped responses.
The two-row horizontal tick labels show physical coordinates and actual marked
ODB frame indices. `F28` means `steps['GMNIA'].frames[28]`, not increment/LPF.
The complete mapping remains in `<job>_pm_response.csv`; the selected P-M frame
is marked by a red diamond. Solver order/reversals are preserved; sparse labels
do not pretend displacement uniquely determines frame number.
`PM_model_response_plots.json` records source CSVs and output paths. Images are
cached by source/selection signature and repaired from CSV if missing; no extra
solve or ODB extraction is required. The combined comparison creates the same
images for no-plate, plated and HSS datasets in its common `diagram` directory.

The same directory also contains `<job>_axial_bending_response.xlsx` for each
requested case/ray. `Axial` and `Bending` export all frames with numeric response
pairs, explicit units and LPF; `Metadata` preserves source paths, solver status
and the selected frame/verification. No extra Python package is required.
Workbook caching is independent of image caching, so a missing/incomplete Excel
file is repaired from CSV without rerendering PNG/SVG or solving again.


## Per-job time limit

Every Step-5 nonlinear job, including P-M jobs for no plates, with plates and
HSS, has a 60-minute wall-clock limit by default. No extra command option is
needed. Each parallel slot has its own clock starting at submission; time
waiting in the queue is excluded. Startup and license waiting after submission
are included.

At the limit the queue requests native Abaqus termination, preserves the saved
results and starts the next job after shutdown is confirmed. Native shutdown
and extraction add a short interval after the solve limit. If shutdown cannot
be confirmed, the queue reports an error instead of reusing a potentially live
solver's slot.

The status is `TIME_LIMIT`, never a verified capacity. Usable saved frames follow
the same P-M selection rule as other incomplete jobs. With no usable frames the
job is explicitly omitted from the plot and the queue continues; no zero or
invented capacity point is created. `--resume` skips these recorded time stops
instead of spending another hour on the same job. To retry intentionally, use
a new run directory. An already running queue must be restarted to load this
source change.

## P-M response selection (response-v4-dual)

For no-plate, plated and HSS jobs, both Perfect and imperfect, the queue and
collector apply the same rule independently to P versus axial shortening and M
versus end rotation. A component with identically zero applied load is excluded.
The earliest accepted event selects BOTH P and M at one actual saved ODB frame;
there is no pairing of independent maxima or interpolated capacity coordinates.

A clear pre-peak knee needs at least 35% persistent tangent reduction, a load
effect of at least 5% of the observed maximum (raised by a six-noise-scale gate),
and strong broken-line evidence over both a straight line and smooth quadratic
(BIC improvement >=10). Each side needs at least four real observations spanning
8% of the loading-branch displacement range. Fits use equal displacement spacing;
dense adaptive increments do not create extra support. Small coordinate jitter
(up to 0.5% of branch span) is ignored for fitting. Real reversals bound the fit;
a later post-peak tail cannot dilute a pre-peak knee. The selected row is the last
observed frame before the fitted onset bracket, whose indices are recorded.

The existing strong softening screen is retained. A recovered load drop is
screened directly in solver order, with >=5% drop/recovery and several actual
increments on each side, so a rotation reversal cannot erase it. These onset
points remain provisional even if the solver completed. If no event is accepted,
the existing verified-peak/ascending-endpoint/attained-maximum fallback remains.
A retracing warning is preserved and cannot certify a provisional point or
replace an earlier accepted event with a later maximum.

`selection_response`, `selection_policy` and per-channel diagnostic evidence are
exported with P-M points; paired response images identify the selected channel,
and each XLSX Metadata sheet records the same policy and evidence. Collection
re-evaluates saved validated curves under this rule without solving them again.
No solver tolerances, run limits, geometry or material properties are changed by
this selection policy; detecting a knee does not establish a design resistance.


### Live CMD progress and per-job response reports

The family/comparison and HSS pipelines relay CAE queue messages to the original
terminal every 15 seconds. Each active job reports elapsed time, converged
step/increment/attempt, LPF and its attained maximum. Adaptive Riks has no known
completion percentage; the queue separately reports processed jobs / total jobs.
Messages are also appended to `STEP5_console_progress.log`.

After each P-M job exits and its ODB is closed/extracted, its solver directory gets
`<job>_axial_bending_response.png`, `.svg` and `.xlsx`. Excel retains all original
frame indices and synchronized P, shortening, M and rotation, including stopped
analyses. `STEP5_response_paths.png/.svg` compares each completed history against
its physical response and stored frame, with at most six jobs per page (`_002`,
etc.). `PM_response_report_index.csv` records status and provenance.

The comparison pipeline also updates `diagram/response_curves` and
`diagram/04_PM_response_paths*.png/.svg` after each job. Existing cached paired
figures/workbooks are reused. All reporting uses saved CSV, never another solve
or an additional ODB extraction. A reporting failure preserves numerical results
and is recorded separately; `--resume` repairs reports without resubmitting an
occupied job. Selection criteria and capacity verification are unchanged.
