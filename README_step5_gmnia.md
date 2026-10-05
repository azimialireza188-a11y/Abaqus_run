# گام ۵: ساخت مدل GMNIA بدون اجرای تحلیل

اسکریپت `abaqus_step5_gmnia.py` فقط CAE مرحلهٔ چهارِ تولیدشده از همین pipeline مرجع را می‌پذیرد و یک فایل CAE و یک INP برای هر مدل انتخاب‌شده تحویل می‌دهد. هیچ Job ارسال نمی‌شود. مش، مختصات دارای نقص، شرایط مرزی و تماس قبلی حفظ می‌شوند. مدل بدون نقصِ مرجع وارد خروجی این مرحله نمی‌شود. Provenance مرحلهٔ چهار داخل Model description خوانده و قبل از هر تغییر با قرارداد `abaqus_complete_model_m20.py` کنترل می‌شود.

## دستور اجرا

در مثال زیر **350 صرفاً نمونهٔ ورودی fy** برحسب MPa است؛ مقدار مناسب مصالح خودتان را وارد کنید. پوشهٔ خروجی باید جدید یا خالی باشد.

```bat
cd /d "D:\CFS-Column\New folder (6)"
abaqus cae noGUI=abaqus_step5_gmnia.py -- ^
  --source-cae "D:\CFS-Column\New folder (7)\Step4_imperfections.cae" ^
  --output-dir "D:\CFS-Column\New folder (7)\Step5_FY350" ^
  --fy 350
```

با این دستور هر هفت مدل `STEP4_*` تبدیل می‌شوند. برای شروع با یک مدل، به دستور اضافه کنید:

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
| `--cpus` | 8 |

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
الگوی حافظهٔ 24000 MB کد مرجع را استفاده می‌کنند. CAE و INP هر مدل از یک Model/Job
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
JSON and a dependency-free SVG curve, and reports the peak load.

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

The SVG is publication-friendly vector output. The CSV can be imported directly
into Excel, Origin, MATLAB or Python for final figure styling.


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

The builder writes `step5_postpeak_stop.for`, an Abaqus/Standard `URDFIL`
routine. A minimal `*NODE FILE, NSET=STEP5_STOP, FREQUENCY=1` request is
injected into every INP so URDFIL is called after every converged increment.
URDFIL tracks the largest LPF reached and sets `LSTOP=1` at the first
post-peak increment satisfying the 70% rule.

The ratio is configurable, but the publication default is 0.70:

```bat
--postpeak-stop-ratio 0.70
```

The builder also writes a sequential batch file such as:

```text
run_step5_queue_70_pct.bat
```

Each Abaqus command in that queue explicitly includes:

```text
user="step5_postpeak_stop.for"
```

and the next job is attempted even if the preceding job exits with an error.

The 70% rule is the primary desired post-peak completion criterion. To keep it
from competing with an unrelated finishing condition, the Step-5 builder no
longer activates a default displacement stop. A displacement limit is used only
when `--max-end-displacement-mm` is supplied explicitly. The default maximum
increment count is increased to 5000 as a high safety cap. A job can still
terminate earlier because of genuine nonconvergence, licensing, or another solver
error; such a case must be reported as not reaching the common post-peak
criterion and should not be treated as a complete publication run.

Abaqus/Standard user-subroutine compilation must be configured on the machine
for the URDFIL-controlled jobs. The CAE jobs store the generated Fortran file as
their user subroutine, and the generated queue passes it explicitly on the
command line.
