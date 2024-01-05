[_English version_](./README.md)


# رگرسیون خطی

### چکیده
این کد  [\(linear_regression.py\)](./linear_regression.py) یک پیاده‌سازی ساده از رگرسیون خطی با استفاده از روش `کمترین مربعات` ارائه می‌دهد. رگرسیون خطی یک الگوریتم یادگیری نظارتی است که برای پیش‌بینی یک متغیر هدف پیوسته بر اساس یک یا چند ویژگی مستقل استفاده می‌شود.

### کلاس: `LinearRegression`

#### مقدمه
```python
class LinearRegression:
    def __init__(self):
        self.theta = None
```
این کلاس با یک خصوصیت خالی به نام `theta` مقداردهی اولیه می‌شود که در نهایت ضرایب مدل رگرسیون خطی را ذخیره خواهد کرد.

#### متد: `fit(X, y)`
```python
    def fit(self, X, y):
        X_augmented = np.column_stack((np.ones(X.shape[0]), X))
        self.theta = np.linalg.lstsq(
            X_augmented.T @ X_augmented, X_augmented.T @ y, rcond=None
        )[0]
```
این متد مدل رگرسیون خطی را به داده‌های ورودی (`X` و `y`) می‌پیوندد. یک ستون تمام یک‌ها به ماتریس ورودی `X` اضافه می‌شود تا به در نظر گرفتن اصطلاح اختلاف مبدا کمک کند. سپس این متد ضرایب (`theta`) را با استفاده از روش کمترین مربعات ارائه شده توسط `numpy.linalg.lstsq` محاسبه می‌کند.

#### متد: `predict(X)`
```python
    def predict(self, X):
        X_augmented = np.column_stack((np.ones(X.shape[0]), X))
        y_predict = X_augmented @ self.theta
        return y_predict
```
متد `predict` ویژگی‌های ورودی `X` را گرفته و متغیر هدف متناظر را با استفاده از ضرایب یادگرفته‌شده پیش‌بینی می‌کند. دوباره، ماتریس ورودی با یک ستون تمام یک‌ها گسترده می‌شود قبل از محاسبه مقادیر پیش‌بینی شده.

#### متد: `score(X, y, threshold=0.5)`
```python
    def score(self, X, y, threshold=0.5):
        y_predict = self.predict(X)
        return np.mean(np.abs(y_predict - y) <= threshold)
```
متد `score` دقت پیش‌بینی‌های مدل را با مقایسه آن‌ها با مقادیر هدف واقعی ارزیابی می‌کند. از یک آستانه مشخص (مقدار پیش‌فرض 0.5) برای تعیین اینکه آیا مقادیر پیش‌بینی صحیح هستند یا خیر، استفاده می‌شود. امتیاز به عنوان میانگین اختلاف‌های مطلق بین مقادیر پیش‌بینی شده و واقعی در داخل آستانه محاسبه می‌شود.

### مفاهیم ریاضی

#### رگرسیون خطی
رگرسیون خطی مدلی است که رابطه بین یک متغیر وابسته (`y`) و یک یا چند متغیر مستقل (`X`) را با مدل کردن یک خط خطی نشان می‌دهد. هدف این است که ضرایب (`theta`) را پیدا کنیم که حداقل اختلاف مربعی بین مقادیر پیش‌بینی شده و واقعی را به دست آوریم.

#### روش کمترین مربعات
روش کمترین مربعات یک روش برای یافتن ضرایب (`theta`) است که حداقل اختلافات مربعی را کمینه می‌کند. در این پیاده‌سازی، از تابع `numpy.linalg.lstsq` برای محاسبه آن استفاده شده است.

### مثال استفاده

می‌توانید مثال کامل را در [./linear_regression.ipynb](./linear_regression.ipynb) مشاهده کنید.

```python
import numpy as np

# داده‌های نمونه
X_train = np.array([[1], [2], [3]])
y_train = np.array([2, 3, 4])

# ایجاد و یادگیری مدل رگرسیون خطی
model = LinearRegression()
model.fit(X_train, y_train)

# پیش‌بینی‌ها
X_test = np.array([[4], [5]])
predictions = model.predict(X_test)

# ارزیابی

score = model.score(X_test, np.array([5, 6]))
print(f"دقت مدل: {score}")
```