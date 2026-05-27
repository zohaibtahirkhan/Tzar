# **FastAPI Detailed Cheatsheet**  

---

## **1. Basic Setup**  

### **Minimal FastAPI App**  
```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Hello World"}
```

### **Run the App**  
```bash
uvicorn main:app --reload
```
- `--reload` enables auto-reload during development.  

---

## **2. Path Parameters & Query Parameters**  

### **Path Parameters**  
```python
@app.get("/items/{item_id}")
def get_item(item_id: int):  # Type hints for automatic validation
    return {"item_id": item_id}
```

### **Query Parameters**  
```python
@app.get("/items/")
def list_items(skip: int = 0, limit: int = 10):  # Default values
    return {"skip": skip, "limit": limit}
```

### **Combining Path & Query Params**  
```python
@app.get("/users/{user_id}/items")
def get_user_items(user_id: int, q: str = None):
    return {"user_id": user_id, "q": q}
```

---

## **3. Request Body (POST, PUT, PATCH)**  

### **Using Pydantic Models**  
```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None

@app.post("/items/")
def create_item(item: Item):
    return {"item": item.dict()}
```

### **Optional Fields & Nested Models**  
```python
class User(BaseModel):
    username: str
    full_name: str | None = None

class Order(BaseModel):
    user: User
    items: list[Item]
```

---

## **4. Response Model & Status Codes**  

### **Custom Response Model**  
```python
@app.post("/items/", response_model=Item)  # Excludes unset fields
def create_item(item: Item):
    return item
```

### **Status Codes**  
```python
from fastapi import status

@app.post("/items/", status_code=status.HTTP_201_CREATED)
def create_item(item: Item):
    return item
```

### **Response Headers & Cookies**  
```python
from fastapi import Response

@app.get("/set-cookie/")
def set_cookie(response: Response):
    response.set_cookie(key="token", value="abc123")
    return {"message": "Cookie set"}
```

---

## **5. Forms, Files, and Uploads**  

### **Form Data**  
```python
from fastapi import Form

@app.post("/login/")
def login(username: str = Form(...), password: str = Form(...)):
    return {"username": username}
```

### **File Upload**  
```python
from fastapi import UploadFile, File

@app.post("/upload/")
def upload_file(file: UploadFile = File(...)):
    return {"filename": file.filename, "content_type": file.content_type}
```

### **Multiple Files**  
```python
@app.post("/upload-files/")
def upload_files(files: list[UploadFile] = File(...)):
    return {"filenames": [f.filename for f in files]}
```

---

## **6. Error Handling**  

### **HTTPException**  
```python
from fastapi import HTTPException

@app.get("/items/{item_id}")
def get_item(item_id: int):
    if item_id == 0:
        raise HTTPException(
            status_code=404,
            detail="Item not found",
            headers={"X-Error": "Item ID invalid"}
        )
    return {"item_id": item_id}
```

### **Custom Exception Handler**  
```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

class UnicornException(Exception):
    def __init__(self, name: str):
        self.name = name

@app.exception_handler(UnicornException)
async def unicorn_exception_handler(request: Request, exc: UnicornException):
    return JSONResponse(
        status_code=418,
        content={"message": f"Oops! {exc.name} did something wrong."},
    )
```

---

## **7. Dependency Injection**  

### **Simple Dependency**  
```python
from fastapi import Depends

def get_db():
    db = "db_connection"
    try:
        yield db
    finally:
        db.close()

@app.get("/items/")
def read_items(db: str = Depends(get_db)):
    return {"db": db}
```

### **Class-based Dependencies**  
```python
class AuthChecker:
    def __init__(self, role: str):
        self.role = role

    def __call__(self, token: str = Header(...)):
        if token != "secret":
            raise HTTPException(status_code=403, detail="Forbidden")
        return self.role

check_admin = AuthChecker("admin")

@app.get("/admin/")
def admin_panel(role: str = Depends(check_admin)):
    return {"role": role}
```

---

## **8. Background Tasks**  

```python
from fastapi import BackgroundTasks

def log_message(message: str):
    with open("log.txt", "a") as f:
        f.write(message + "\n")

@app.post("/notify/")
def send_notification(email: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(log_message, f"Email sent to {email}")
    return {"message": "Notification sent"}
```

---

## **9. Middleware & CORS**  

### **Adding Middleware**  
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (restrict in production)
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)
```

### **Custom Middleware**  
```python
from fastapi import Request

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Process-Time"] = "123ms"
    return response
```

---

## **10. Testing with TestClient**  

```python
from fastapi.testclient import TestClient

client = TestClient(app)

def test_read_item():
    response = client.get("/items/42")
    assert response.status_code == 200
    assert response.json() == {"item_id": 42}

def test_create_item():
    response = client.post("/items/", json={"name": "Foo", "price": 45.0})
    assert response.status_code == 201
```

---

## **11. WebSockets**  

```python
from fastapi import WebSocket

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"Message: {data}")
```

---

## **12. Static Files**  

```python
from fastapi.staticfiles import StaticFiles

app.mount("/static", StaticFiles(directory="static"), name="static")
```

---

## **13. OpenAPI/Swagger & Redoc**  
- **Interactive Docs**: `http://localhost:8000/docs` (Swagger UI)  
- **Alternative Docs**: `http://localhost:8000/redoc`  

### **Customizing OpenAPI**  
```python
app = FastAPI(
    title="My API",
    description="API for my project",
    version="1.0.0",
    openapi_tags=[{
        "name": "users",
        "description": "Operations with users",
    }]
)
```

---

This cheatsheet covers **FastAPI** essentials and advanced features. For more details, check the **[official FastAPI docs](https://fastapi.tiangolo.com/)**. 🚀