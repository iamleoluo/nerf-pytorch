// Vue 應用
const { createApp } = Vue;

const app = createApp({
    data() {
        return {
            cameras: [],
            selectedCamera: null,
            currentImage: '',
            validationResults: [],
            scene: null,
            camera: null,
            renderer: null,
            cameraObjects: [],
            showColmap: true,
            showNerf: true,
            clickHandlerAdded: false
        }
    },
    mounted() {
        this.initThreeJS();
        // 自動載入數據
        this.loadData();
        // 自動執行驗證
        this.validateData();
    },
    methods: {
        // 初始化 Three.js
        initThreeJS() {
            const container = document.getElementById('camera-visualization');
            
            // 創建場景
            this.scene = new THREE.Scene();
            this.scene.background = new THREE.Color(0xf0f0f0);

            // 創建相機
            this.camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
            this.camera.position.set(5, 5, 5);
            this.camera.lookAt(0, 0, 0);

            // 創建渲染器
            this.renderer = new THREE.WebGLRenderer({ antialias: true });
            this.renderer.setSize(container.clientWidth, container.clientHeight);
            container.appendChild(this.renderer.domElement);

            // 添加軌道控制器
            const controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;

            // 添加坐標軸
            const axesHelper = new THREE.AxesHelper(5);
            this.scene.add(axesHelper);

            // 添加網格
            const gridHelper = new THREE.GridHelper(10, 10);
            this.scene.add(gridHelper);

            // 動畫循環
            const animate = () => {
                requestAnimationFrame(animate);
                controls.update();
                this.renderer.render(this.scene, this.camera);
            };
            animate();

            // 窗口大小調整
            window.addEventListener('resize', () => {
                this.camera.aspect = container.clientWidth / container.clientHeight;
                this.camera.updateProjectionMatrix();
                this.renderer.setSize(container.clientWidth, container.clientHeight);
            });
        },

        // 載入數據
        async loadData() {
            try {
                const response = await fetch('/api/cameras');
                const data = await response.json();
                
                if (data.status === 'success') {
                    this.cameras = data.data;
                    console.log(`載入了 ${this.cameras.length} 個相機`);
                    this.visualizeCameras();
                } else {
                    throw new Error(data.message);
                }
            } catch (error) {
                console.error('載入數據失敗:', error);
                alert('載入數據失敗: ' + error.message);
            }
        },

        // 可視化相機
        visualizeCameras() {
            console.log(`開始可視化 ${this.cameras.length} 個相機`);
            
            // 清除現有的相機對象
            this.cameraObjects.forEach(obj => this.scene.remove(obj));
            this.cameraObjects = [];

            // 創建相機可視化
            this.cameras.forEach((camera, index) => {
                console.log(`處理相機 ${index + 1}: ${camera.file_path}`);
                
                // NeRF 相機
                if (this.showNerf && camera.transform_matrix) {
                    try {
                        const nerfMatrix = new THREE.Matrix4().fromArray(camera.transform_matrix.flat());
                        const nerfPosition = new THREE.Vector3();
                        const nerfQuaternion = new THREE.Quaternion();
                        const nerfScale = new THREE.Vector3();
                        nerfMatrix.decompose(nerfPosition, nerfQuaternion, nerfScale);

                        // 創建 NeRF 相機模型
                        const nerfGeometry = new THREE.ConeGeometry(0.1, 0.3, 8);
                        const nerfMaterial = new THREE.MeshBasicMaterial({ color: 0xff0000 });
                        const nerfCameraMesh = new THREE.Mesh(nerfGeometry, nerfMaterial);
                        
                        nerfCameraMesh.position.copy(nerfPosition);
                        nerfCameraMesh.quaternion.copy(nerfQuaternion);
                        nerfCameraMesh.rotateX(Math.PI / 2);
                        nerfCameraMesh.userData = { camera, type: 'nerf' };
                        
                        this.scene.add(nerfCameraMesh);
                        this.cameraObjects.push(nerfCameraMesh);
                        
                        console.log(`NeRF相機 ${camera.file_path} 位置:`, nerfPosition);
                    } catch (error) {
                        console.error(`NeRF相機 ${camera.file_path} 處理錯誤:`, error);
                    }
                }

                // COLMAP 相機
                if (this.showColmap && camera.colmap_transform) {
                    try {
                        const colmapMatrix = new THREE.Matrix4().fromArray(camera.colmap_transform.flat());
                        const colmapPosition = new THREE.Vector3();
                        const colmapQuaternion = new THREE.Quaternion();
                        const colmapScale = new THREE.Vector3();
                        colmapMatrix.decompose(colmapPosition, colmapQuaternion, colmapScale);

                        // 創建 COLMAP 相機模型
                        const colmapGeometry = new THREE.ConeGeometry(0.1, 0.3, 8);
                        const colmapMaterial = new THREE.MeshBasicMaterial({ color: 0x0000ff });
                        const colmapCameraMesh = new THREE.Mesh(colmapGeometry, colmapMaterial);
                        
                        colmapCameraMesh.position.copy(colmapPosition);
                        colmapCameraMesh.quaternion.copy(colmapQuaternion);
                        colmapCameraMesh.rotateX(Math.PI / 2);
                        colmapCameraMesh.userData = { camera, type: 'colmap' };
                        
                        this.scene.add(colmapCameraMesh);
                        this.cameraObjects.push(colmapCameraMesh);
                        
                        console.log(`COLMAP相機 ${camera.file_path} 位置:`, colmapPosition);
                    } catch (error) {
                        console.error(`COLMAP相機 ${camera.file_path} 處理錯誤:`, error);
                    }
                }
            });

            console.log(`總共創建了 ${this.cameraObjects.length} 個相機對象`);

            // 添加點擊事件（只添加一次）
            if (!this.clickHandlerAdded) {
                this.addCameraClickHandler();
                this.clickHandlerAdded = true;
            }
        },

        // 添加相機點擊處理器
        addCameraClickHandler() {
            const raycaster = new THREE.Raycaster();
            const mouse = new THREE.Vector2();

            this.renderer.domElement.addEventListener('click', (event) => {
                // 計算鼠標位置
                const rect = this.renderer.domElement.getBoundingClientRect();
                mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
                mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

                // 發射射線
                raycaster.setFromCamera(mouse, this.camera);
                const intersects = raycaster.intersectObjects(this.cameraObjects);

                if (intersects.length > 0) {
                    const selected = intersects[0].object;
                    console.log('選中相機:', selected.userData.camera.file_path);
                    this.selectCamera(selected.userData.camera);
                }
            });
        },

        // 選擇相機
        selectCamera(camera) {
            console.log('選擇相機:', camera.file_path);
            this.selectedCamera = camera;
            this.currentImage = `/api/image/${camera.file_path}`;
            console.log('圖像URL:', this.currentImage);
        },

        // 驗證數據
        async validateData() {
            try {
                const response = await fetch('/api/validate');
                const data = await response.json();
                
                if (data.status === 'success') {
                    this.validationResults = data.results;
                } else {
                    throw new Error(data.message);
                }
            } catch (error) {
                console.error('驗證數據失敗:', error);
                alert('驗證數據失敗: ' + error.message);
            }
        },

        // 導出報告
        exportReport() {
            // 創建報告內容
            const report = {
                title: 'NeRF 數據驗證報告',
                timestamp: new Date().toISOString(),
                results: this.validationResults,
                camera_count: this.cameras.length
            };

            // 轉換為 JSON 字符串
            const reportStr = JSON.stringify(report, null, 2);

            // 創建下載鏈接
            const blob = new Blob([reportStr], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'nerf_validation_report.json';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        },

        // 格式化位置信息
        formatPosition(matrix) {
            const position = new THREE.Vector3();
            const quaternion = new THREE.Quaternion();
            const scale = new THREE.Vector3();
            new THREE.Matrix4().fromArray(matrix).decompose(position, quaternion, scale);
            
            return `[${position.x.toFixed(3)}, ${position.y.toFixed(3)}, ${position.z.toFixed(3)}]`;
        },

        // 格式化旋轉信息
        formatRotation(matrix) {
            const position = new THREE.Vector3();
            const quaternion = new THREE.Quaternion();
            const scale = new THREE.Vector3();
            new THREE.Matrix4().fromArray(matrix).decompose(position, quaternion, scale);
            
            // 將四元數轉換為歐拉角（度）
            const euler = new THREE.Euler().setFromQuaternion(quaternion);
            const x = (euler.x * 180 / Math.PI).toFixed(1);
            const y = (euler.y * 180 / Math.PI).toFixed(1);
            const z = (euler.z * 180 / Math.PI).toFixed(1);
            
            return `[${x}°, ${y}°, ${z}°]`;
        },

        // 計算位置差異
        calculatePositionDifference(matrix1, matrix2) {
            const pos1 = new THREE.Vector3();
            const pos2 = new THREE.Vector3();
            
            new THREE.Matrix4().fromArray(matrix1).decompose(pos1, new THREE.Quaternion(), new THREE.Vector3());
            new THREE.Matrix4().fromArray(matrix2).decompose(pos2, new THREE.Quaternion(), new THREE.Vector3());
            
            const distance = pos1.distanceTo(pos2);
            return distance.toFixed(4);
        },

        // 獲取警告樣式
        getAlertClass(status) {
            switch (status) {
                case 'success': return 'alert-success';
                case 'warning': return 'alert-warning';
                case 'error': return 'alert-danger';
                default: return 'alert-info';
            }
        },

        // 切換顯示 COLMAP 相機
        toggleColmap() {
            this.showColmap = !this.showColmap;
            this.visualizeCameras();
        },

        // 切換顯示 NeRF 相機
        toggleNerf() {
            this.showNerf = !this.showNerf;
            this.visualizeCameras();
        }
    }
});

// 掛載 Vue 應用
app.mount('#app'); 