# گام ۵: ساخت مدل GMNIA بدون اجرای تحلیل

اسکریپت `abaqus_step5_gmnia.py` فقط CAE مرحلهٔ چهارِ تولیدشده از همین pipeline مرجع را می‌پذیرد و یک فایل CAE و یک INP برای هر مدل انتخاب‌شده تحویل می‌دهد. هیچ Job ارسال نمی‌شود. مش، مختصات دارای نقص، شرایط مرزی و تماس قبلی حفظ می‌شوند. مدل بدون نقصِ مرجع نیز به‌طور پیش‌فرض وارد خروجی این مرحله می‌شود. Provenance مرحلهٔ چهار داخل Model description خوانده و قبل از هر تغییر با قرارداد `abaqus_complete_model_m20.py` کنترل می‌شود.

## دستور اجرا

### نمایش مرحلهٔ جاری در CMD

ساخت مدل و اجرای صف اکنون پیام‌های فوری همراه ساعت چاپ می‌کنند. در اجرای
`run_step5_queue_70_pct.bat`، نام و شمارهٔ مدل (`JOB 1/8`)، ارسال حل (`SUBMIT`)،
حل جاری (`SOLVING`)، خروج حلگر (`SOLVER EXIT`)، استخراج (`EXTRACT`)، رسم نمودار
(`PLOT`) و مقایسه (`COMPARISON`) مشخص می‌شوند.

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
- مصالح الاستیک–کاملاً پلاستیک با fy ورودی؛ E و ضریب پواسون قبلی حفظ می‌شوند. شکست یا آسیب مصالح و پیچ‌ها تعریف نشده است.
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
| `--max-increments` | 1000 |
| `--max-end-displacement-mm` | 0.01L؛ برای L=3600 برابر 36 mm |
| `--field-frequency` | 1؛ هر Increment |
| `--cpus` | همهٔ CPUهای در دسترس |

شرط جابه‌جایی مربوط به **U3 مثبت یک گره در انتهای پایین** است؛ کوتاه‌شدگی کل ستون نیست. گره پایش در Set به نام `STEP5_STOP` مشخص است. اندازهٔ مش از مرحلهٔ چهار می‌آید و اینجا تغییر نمی‌کند. طول‌قوس، زمان فیزیکی تحلیل نیست. معیارهای تعادل پیش‌فرض Abaqus ضعیف نشده‌اند و پایدارسازی مصنوعی اضافه نشده است.

این‌ها تنظیمات اولیه‌اند؛ ساخت موفق فایل‌ها، تضمین همگرایی یا صحت پژوهشی تحلیل نیست. پس از حل باید ثبت بار اوج و بخش کافی از شاخهٔ نزولی، تعادل، رفتار تماس و حساسیت عددی بررسی شود. پایان به‌دلیل سقف Increment را خودکار «ظرفیت نهایی به‌دست آمد» تلقی نکنید.

## استخراج نتایج در مرحلهٔ بعد

در توضیح هر Model، تنظیمات و `reference_force_N_per_end` ثبت می‌شوند:

`P = LPF * reference_force_N_per_end`

این نیرو مربوط به **یک انتها** است. نیروهای دو انتها را با هم جمع نکنید و واکنش مهار میانی را بار ستون نگیرید. سطح مرجع از لبه‌های مش S4R محاسبه می‌شود و ممکن است اندکی با مساحت هندسی اسمی تفاوت داشته باشد.

کوتاه‌شدگی سازگار با بارگذاری توزیع‌شده:

`delta = area_weighted_mean(U3_bottom) - area_weighted_mean(U3_top)`

وزن‌ها در `end_area_weights` توضیح Model ذخیره می‌شوند. History خروجی جابه‌جایی دو انتها و نیرو/گشتاور تمام پیچ‌ها در هر Increment موجود است؛ بنابراین نیروی پیچ‌ها در Increment بار اوج قابل بازیابی خواهد بود. کد حاضر فقط مدل می‌سازد و نمودار یا ODB تولید نمی‌کند.

فایل CAE همهٔ مدل‌ها و Jobهای آماده را دارد؛ هر INP به یک Job تعلق دارد. پوشهٔ خروجی فقط CAE و INP دریافت می‌کند. فایل‌های شروع خود Abaqus مانند `abaqus.rpy` ممکن است در پوشهٔ اجرای دستور ایجاد شوند.


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

## Repair an existing queue after the ifort error

Pull the latest repository. In the directory containing your existing eight
`STEP5_*.inp` files and `Step5_GMNIA_FY240.cae`, run:

```bat
abaqus cae noGUI="C:\Users\810200014.HAMI.000\Documents\Abaqus_run\abaqus_step5_queue_nogui.py" -- --run-dir . --repair-only
run_step5_queue_70_pct.bat
```

Repair backs up the previous BAT, validates the CAE metadata and rewrites the
queue; it does not rebuild/modify CAE or INP and does not submit jobs. The new BAT
references the repository runner by absolute path. Keep that repository path
available. Existing ODBs and live locks are never overwritten/deleted by this
runner. Residual solver files from failed compilation are moved into a unique
per-job backup directory before submission, preserving their diagnostics. For completed/failed existing ODBs, preserve results and replot using:

```bat
abaqus cae noGUI="C:\Users\810200014.HAMI.000\Documents\Abaqus_run\abaqus_step5_queue_nogui.py" -- --run-dir . --extract-only
```

If several STEP5 CAEs are present, specify the exact `--cae` path. Use `--jobs`
to restrict the exact jobs intentionally. A failed earlier job does not prevent
attempting the remaining queue, except that preflight conflicts stop submission
before any job starts. Solver stdout/stderr is saved per job as
`STEP5_*_queue_solver.log`.

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
Standard element operations. Explicit --cpus/--gpus overrides remain available
for deliberate use; omit old --cpus 8 commands to use the automatic default.
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
