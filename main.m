clear; clc;

% Membaca dataset dari file CSV
data = dlmread('DataSet_KualitasUdara_K1.csv', ',', 1, 0);
% Mengekstrak  data (kolom 2 sampai 97)
A = data(:, 2:end);

%% BAGIAN 1
% Menampilkan ukuran matriks diawal untuk memverifikasi 
fprintf('BAGIAN 1: Matriks A dan Normalisasi Z-Score\n');
fprintf('Ukuran matriks A: %d baris x %d kolom\n', size(A, 1), size(A, 2));
[m, n] = size(A); % m = 365 (baris), n = 96 (kolom)

mean_j = (ones(1,m) * A) / m; % menghitung mean dari setiap kolom menggunakan operasi
                              % matriks dengan menjumlahkan semua baris tiap kolom, lalu dibagi m
simpangan = A - ones(m,1) * mean_j; % Menghtung simpangan agar rata-rata tiap kolom = 0
varians = (ones(1,m) * (simpangan .^ 2)) / m; % Menghitung varians setiap kolom
stddv = sqrt(varians); % Menghitung standar deviasi
% Mulai menghitung z_Score
Z = simpangan ./ (ones(m,1) * stddv);
fprintf('Ukuran matriks Z (setelah normalisasi): %d baris x %d kolom\n', size(Z,1), size(Z,2));
% verifikasi hasil normalisasi
mean_Z    = (ones(1,m) * Z) / m;
varians_Z   = (ones(1,m) * ((Z - ones(m,1)*mean_Z) .^ 2)) / m;
stddv_Z = sqrt(varians_Z);

fprintf('Verifikasi Normalisasi Z-Score:\n');
fprintf('Max nilai absolut mean Z : %.2e\n', max(abs(mean_Z)));
fprintf('Max selisih std Z dari 1 : %.2e\n', max(abs(stddv_Z - 1)));
fprintf('Min nilai Z : %.4f\n', min(Z(:)));
fprintf('Max nilai Z : %.4f\n', max(Z(:)));

%% Bagian 2 
% Melakukan SVD penuh pada matriks Z = U * S * V'
fprintf('BAGIAN 2: SVD Penuh dan Analisis Spektrum\n');

% Menghasilkan U(365x96), S(96x96), V(96x96)
[U, S, V] = svd(Z, 'econ');

% Menampilkan ukuran ketiga matriks hasil SVD untuk verifikasi
fprintf('Ukuran U  : %d x %d\n', size(U,1), size(U,2)); % vektor singular kiri
fprintf('Ukuran S  : %d x %d\n', size(S,1), size(S,2)); % nilai singular
fprintf('Ukuran V  : %d x %d\n', size(V,1), size(V,2)); % vektor singular kanan

% Verifikasi rekonstruksi Z = U * S * V'
Z_rekon = U * S * V';
err_rekon = norm(Z - Z_rekon, 'fro');
fprintf('Verifikasi rekonstruksi Z = U*S*V :\n');
fprintf('norm(Z - U*S*V) = %.2e\n', err_rekon);

% Verifikasi sifat ortogonal U dan V
err_U = max(max(abs(U'*U - eye(size(U,2)))));
err_V = max(max(abs(V'*V - eye(size(V,2)))));
fprintf('Cek U x U = I : max error = %.2e\n', err_U);
fprintf('Cek V x V = I : max error = %.2e\n', err_V);

% Mengambil nilai singular dari diagonal matriks S
sv = diag(S);

% Menghitung energi tiap komponen
energi = sv .^ 2;
total_energi = (ones(1, length(sv)) * energi);
energi_prop = energi / total_energi; % proporsi energi tiap komponen
energi_kum = cumsum(energi_prop); % energi kumulatif

% Menampilkan 10 nilai singular terbesar
fprintf('10 Nilai Singular Terbesar:\n');
fprintf('  i  |  sigma_i  |  Energi(%%)  |  Kumulatif(%%)\n');
fprintf('-----|-----------|------------|-------------\n');
for i = 1:10
    fprintf('  %2d | %9.4f | %10.4f | %11.4f\n', ...
        i, sv(i), energi_prop(i)*100, energi_kum(i)*100);
end

% Dekomposisi rank-1: Z = sigma1*u1*v1' + sigma2*u2*v2' + ...
fprintf('Dekomposisi Rank-1 (20 komponen pertama):\n');
fprintf('  i  |  sigma_i  |  Error sisa\n');
fprintf('-----|-----------|------------\n');

Z_approx = zeros(m, n); % matriks akumulasi komponen rank-1
for i = 1:20
    % Menambahkan komponen rank-1 ke-i
    komponen_i = S(i,i) * U(:,i) * V(:,i)';
    Z_approx = Z_approx + komponen_i;
    sisa = norm(Z - Z_approx, 'fro'); % error sisa setelah i komponen
    fprintf('  %2d | %9.4f | %11.4f\n', i, sv(i), sisa);
end

% Menentukan rank efektif k* berdasarkan energi kumulatif >= 95%
% k* = jumlah komponen minimum yang merangkum 95% variansi data
fprintf('Rank efektif untuk berbagai threshold energi:\n');
for thr = [0.90, 0.95, 0.99]
    k_thr = find(energi_kum >= thr, 1, 'first');
    fprintf('k untuk %.0f%% energi: %d (dari %d total)\n', thr*100, k_thr, n);
end

% Menetapkan rank efektif utama = 95%
k_efektif = find(energi_kum >= 0.95, 1, 'first');
fprintf('Rank efektif k* = %d\n', k_efektif);
fprintf('Energi kumulatif pada k=%d : %.4f%%\n', k_efektif, energi_kum(k_efektif)*100);

% Rekonstruksi rank-k untuk beberapa nilai k
k_list = unique([3, 5, 10, 20, k_efektif]);
fprintf('Rekonstruksi Rank-k:\n');
fprintf('  k  |  Energi Kum(%%)  |  Rel. Error(%%)\n');
fprintf('-----|----------------|---------------\n');
for idx = 1:length(k_list)
    k = k_list(idx);
    % Merekonstruksi Z dengan k komponen pertama
    Z_k = U(:,1:k) * S(1:k,1:k) * V(:,1:k)';
    % Menghitung relative error hasil rekonstruksi
    rel_err = norm(Z - Z_k, 'fro') / norm(Z, 'fro');
    fprintf('  %2d  | %14.4f | %13.4f\n', k, energi_kum(k)*100, rel_err*100);
end

% Menampilkan interpretasi nilai singular
fprintf('Interpretasi Nilai Singular:\n');
fprintf('sigma_1 = %.4f (%.2f%%) -> pola polusi dominan\n', sv(1), energi_prop(1)*100);
fprintf('sigma_2 = %.4f (%.2f%%) -> pola sekunder\n', sv(2), energi_prop(2)*100);
fprintf('sigma_%d = %.4f (%.2f%%) -> mulai merepresentasikan noise\n', ...
    k_efektif+1, sv(k_efektif+1), energi_prop(k_efektif+1)*100);

% Visualisasi spektrum nilai singular dan energi kumulatif
figure;

% Plot spektrum nilai singular
subplot(2, 2, 1);
plot(1:n, sv, 'b-o', 'MarkerSize', 4);
xlabel('Indeks ke-i');
ylabel('sigma_i');
title('Spektrum Nilai Singular');
grid on;

% Plot energi kumulatif dengan garis threshold 95%
subplot(2, 2, 2);
plot(1:n, energi_kum*100, 'r-', 'LineWidth', 2);
hold on;
plot([1 n], [95 95], 'k--', 'LineWidth', 1.5);
plot([k_efektif k_efektif], [0 100], 'g--', 'LineWidth', 1.5);
xlabel('Jumlah komponen k');
ylabel('Energi kumulatif (%)');
title('Energi Kumulatif vs k');
legend('Energi kumulatif', '95% threshold', sprintf('k*=%d', k_efektif));
grid on;

% Plot mode temporal u1 (pola polusi dominan sepanjang 365 hari)
subplot(2, 2, 3);
plot(1:365, U(:,1), 'b-');
xlabel('Hari ke-');
ylabel('Amplitudo');
title('Mode Temporal u1');
grid on;

% Plot mode fitur v1 
subplot(2, 2, 4);
bar(1:n, V(:,1));
xlabel('Indeks Sensor (1-96)');
ylabel('Koefisien v1');
title('Mode Fitur v1');
grid on;

%% Bagian 3 
% Rekonstruksi Matriks dengan Rank Tereduksi
fprintf('\nBAGIAN 3: Rekonstruksi Matriks (k = 3, 5, 10, 20)\n');

% Daftar nilai k yang akan dievaluasi
k_values = [3, 5, 10, 20];
errors = zeros(length(k_values), 1);

% Rumus: Akar dari jumlah kuadrat seluruh elemen matriks
kuadrat_Z = Z .^ 2;
jumlah_kuadrat_Z = sum(sum(kuadrat_Z)); % menjumlahkan semua elemen baris dan kolom
norm_Z_manual = sqrt(jumlah_kuadrat_Z);

for i = 1:length(k_values)
    k_val = k_values(i);

    % Mengambil k kolom/baris pertama dari U, S, dan V
    Uk = U(:, 1:k_val);
    Sk = S(1:k_val, 1:k_val);
    Vk = V(:, 1:k_val);

    % Rekonstruksi matriks tereduksi (Z_k)
    Zk = Uk * Sk * Vk';
    
    % 2. Menghitung relative reconstruction error 
    % Menghitung matriks selisih (Z - Zk)
    matriks_selisih = Z - Zk;

    % Menghitung Frobenius norm dari matriks selisih
    kuadrat_selisih = matriks_selisih .^ 2;
    jumlah_kuadrat_selisih = sum(sum(kuadrat_selisih));
    norm_selisih_manual = sqrt(jumlah_kuadrat_selisih);

    % Menghitung error relatif: ||Z - Z_k||_F / ||Z||_F
    rel_error = norm_selisih_manual / norm_Z_manual;
    errors(i) = rel_error;

    fprintf('Relative reconstruction error untuk k = %2d: %.4f (%.2f%%)\n', k_val, rel_error, rel_error * 100);
end

% Visualisasi Error Rekonstruksi 
figure('Name', 'Reconstruction Error', 'NumberTitle', 'off');
plot(k_values, errors * 100, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
title('Relative Reconstruction Error vs k');
xlabel('Nilai k (Rank Tereduksi)');
ylabel('Relative Error (%)');
xticks(k_values);
grid on;

%% Bagian 4
fprintf('\nBAGIAN 4: Aproksimasi Deret Taylor PM2.5\n');


% data(:,1) adalah kolom 'Hari ke-', A(:,1) adalah data PM2.5_1 (kolom pertama polutan)
x_data = data(:, 1);       
y_data = A(:, 1);       

% Menghapus baris yang mengandung NaN untuk mencegah error kalkulasi matriks
valid_idx = ~isnan(x_data) & ~isnan(y_data);
x_data = x_data(valid_idx);
y_data = y_data(valid_idx);

% Penentuan Titik Puncak
[max_val, max_idx] = max(y_data);
x0 = x_data(max_idx);      

% Pembentukan Window Data (15 Hari sebelum dan sesudah puncak)
win_size = 15;
idx_start = max(1, max_idx - win_size);
idx_end = min(length(x_data), max_idx + win_size);

x_win = x_data(idx_start:idx_end);
y_win = y_data(idx_start:idx_end);

% TRANSFORMASI TITIK PUSAT (Menghindari "Badly Conditioned Polynomial")
dx = x_win - x0; 

% Curve Fitting pada Data Terpusat
p = polyfit(dx, y_win, 8);

% Eksekusi Turunan Analitik
p1 = polyder(p); p2 = polyder(p1); p3 = polyder(p2);
p4 = polyder(p3); p5 = polyder(p4); p6 = polyder(p5);
p7 = polyder(p6);

% Evaluasi di titik pusat lokal (dx = 0)
d0 = polyval(p, 0);  d1 = polyval(p1, 0); d2 = polyval(p2, 0);
d3 = polyval(p3, 0); d4 = polyval(p4, 0); d5 = polyval(p5, 0);
d6 = polyval(p6, 0); d7 = polyval(p7, 0);

% Substitusi Deret Taylor (Orde 3, 5, 7)
T3 = d0 + d1.*dx + (d2/factorial(2)).*dx.^2 + (d3/factorial(3)).*dx.^3;
T5 = T3 + (d4/factorial(4)).*dx.^4 + (d5/factorial(5)).*dx.^5;
T7 = T5 + (d6/factorial(6)).*dx.^6 + (d7/factorial(7)).*dx.^7;

% Kalkulasi Galat Absolut Analitik
f_true = polyval(p, dx);
err3 = abs(f_true - T3);
err5 = abs(f_true - T5);
err7 = abs(f_true - T7);

% Visualisasi Render
figure('Name', 'Aproksimasi Deret Taylor PM2.5', 'NumberTitle', 'off');
plot(x_win, y_win, 'ko', 'MarkerFaceColor', 'k', 'DisplayName', 'Data Asli'); hold on;
plot(x_win, f_true, 'k-', 'LineWidth', 2, 'DisplayName', 'Fungsi Asli f(x)');
plot(x_win, T3, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Taylor Orde 3');
plot(x_win, T5, 'g-.', 'LineWidth', 1.5, 'DisplayName', 'Taylor Orde 5');
plot(x_win, T7, 'b:', 'LineWidth', 1.5, 'DisplayName', 'Taylor Orde 7');
plot(x0, max_val, 'm*', 'MarkerSize', 10, 'DisplayName', 'Titik Puncak');
legend('Location', 'best'); xlabel('Hari Ke-'); ylabel('Konsentrasi PM2.5'); grid on;

figure('Name', 'Analisis Galat Aproksimasi', 'NumberTitle', 'off');
plot(x_win, err3, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Galat Orde 3'); hold on;
plot(x_win, err5, 'g-', 'LineWidth', 1.5, 'DisplayName', 'Galat Orde 5');
plot(x_win, err7, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Galat Orde 7');
legend('Location', 'best'); xlabel('Hari Ke-'); ylabel('Galat Absolut'); grid on;



%% Bagian 5
fprintf('BAGIAN 5: Perhitungan Paparan Polutan Menggunakan Integral\n');
%Turunan parsial fungsi konsentrasi polutan
PM25 = A(:, 1); % PM2.5 titik pemantauan 1
n_hari = length(PM25); % 365
%Suhu dalam Celsius: 25-35 derajat
T = 28 + 5 * sin(2*pi*(1:n_hari)/365)';
%Kelembaban dalam persen: 60-90%
H = 75 + 15 * cos(2*pi*(1:n_hari)/365)';
%Kecepatan Angin dalam m/s: 1-5 m/s
W = 3 + 2 * sin(2*pi*(1:n_hari)/180)';

%Rumus: C(T, H, W) = a0 + a1*T + a2*H + a3*W + a4*T*H + a5*H*W
fprintf('\nMenghitung koefisien regresi...\n');

%Susun matriks fitur X
X = [ones(n_hari,1), T, H, W, T.*H, H.*W];

%Hitung koefisien dengan metode Least Squares: a = (X'X)^-1 * X' * C
a = (X' * X) \ (X' * PM25);

fprintf('Koefisien model berhasil dihitung.\n');
fprintf('  a0 (konstanta) = %.4f\n', a(1));
fprintf('  a1 (suhu T)    = %.4f\n', a(2));
fprintf('  a2 (kelembaban H) = %.4f\n', a(3));
fprintf('  a3 (angin W)   = %.4f\n', a(4));
fprintf('  a4 (T*H)       = %.4f\n', a(5));
fprintf('  a5 (H*W)       = %.4f\n', a(6));

fprintf('\n Turunan parsial analitik \n'); 

%Hitung nilai rata-rata T, H, W
T_mean = mean(T);
H_mean = mean(H);
W_mean = mean(W);

dC_dT = a(2) + a(5) * H_mean;
dC_dH = a(3) + a(5) * T_mean + a(6) * W_mean;
dC_dW = a(4) + a(6) * H_mean;

fprintf('Dihitung pada titik rata-rata: T=%.2f, H=%.2f, W=%.2f\n', T_mean, H_mean, W_mean);
fprintf('\ndC/dT (terhadap suhu)        = %.4f ug/m3 per degC\n', dC_dT);
fprintf('dC/dH (terhadap kelembaban)  = %.4f ug/m3 per persen\n', dC_dH);
fprintf('dC/dW (terhadap angin)       = %.4f ug/m3 per m/s\n', dC_dW);

%Rumus beda hingga: df/dx ≈ [f(x+h) - f(x-h)] / (2h)

fprintf('\n Verifikasi turunan numerik\n');

h = 0.01; % langkah kecil

% Fungsi anonim model C
C_model = @(t, hh, w) a(1) + a(2)*t + a(3)*hh + a(4)*w + a(5)*t.*hh + a(6)*hh.*w;

%Turunan numerik di titik rata-rata
dC_dT_num = (C_model(T_mean+h, H_mean, W_mean) - C_model(T_mean-h, H_mean, W_mean)) / (2*h);
dC_dH_num = (C_model(T_mean, H_mean+h, W_mean) - C_model(T_mean, H_mean-h, W_mean)) / (2*h);
dC_dW_num = (C_model(T_mean, H_mean, W_mean+h) - C_model(T_mean, H_mean, W_mean-h)) / (2*h);

fprintf('dC/dT numerik = %.4f  (analitik = %.4f)\n', dC_dT_num, dC_dT);
fprintf('dC/dH numerik = %.4f  (analitik = %.4f)\n', dC_dH_num, dC_dH);
fprintf('dC/dW numerik = %.4f  (analitik = %.4f)\n', dC_dW_num, dC_dW);

%Hitung turunan parsial harian untuk semua data
dC_dT_harian = a(2) + a(5) .* H;
dC_dH_harian = a(3) + a(5) .* T + a(6) .* W;
dC_dW_harian = a(4) + a(6) .* H;

%visualisasi 
figure('Name', 'Bagian 5 - Turunan Parsial', 'Position', [100, 100, 1200, 800]);

%Plot 1: Data PM2.5 asli
subplot(2, 3, 1);
plot(1:n_hari, PM25, 'b-', 'LineWidth', 1.2);
xlabel('Hari ke-'); ylabel('PM2.5 (ug/m3)');
title('Data PM2.5 Asli (Titik 1)');
grid on;

%Plot 2: dC/dT harian
subplot(2, 3, 2);
plot(1:n_hari, dC_dT_harian, 'r-', 'LineWidth', 1.2);
xlabel('Hari ke-'); ylabel('dC/dT');
title('Turunan Parsial terhadap Suhu (T)');
grid on;
yline(0, 'k--', 'nol');

%Plot 3: dC/dH harian
subplot(2, 3, 3);
plot(1:n_hari, dC_dH_harian, 'g-', 'LineWidth', 1.2);
xlabel('Hari ke-'); ylabel('dC/dH');
title('Turunan Parsial terhadap Kelembaban (H)');
grid on;
yline(0, 'k--', 'nol');

%Plot 4: dC/dW harian
subplot(2, 3, 4);
plot(1:n_hari, dC_dW_harian, 'm-', 'LineWidth', 1.2);
xlabel('Hari ke-'); ylabel('dC/dW');
title('Turunan Parsial terhadap Kecepatan Angin (W)');
grid on;
yline(0, 'k--', 'nol');

%Plot 5: Membandingkan turunan
subplot(2, 3, 5);
bar([dC_dT, dC_dH, dC_dW]);
set(gca, 'XTickLabel', {'dC/dT', 'dC/dH', 'dC/dW'});
ylabel('Nilai Turunan di Titik Rata-rata');
title('Perbandingan Turunan Parsial');
grid on;

%Plot 6: PM2.5 vs Suhu untuk scatter plyt
subplot(2, 3, 6);
scatter(T, PM25, 10, 'b', 'filled', 'MarkerFaceAlpha', 0.3);
xlabel('Suhu (degC)'); ylabel('PM2.5 (ug/m3)');
title('Hubungan PM2.5 vs Suhu');
grid on;

sgtitle('Bagian 5: Analisis Turunan Parsial Konsentrasi Polutan PM2.5');

fprintf('\n=== INTERPRETASI FISIK ===\n');

if dC_dT < 0
    fprintf('dC/dT < 0 : Ketika suhu naik 1 degC, konsentrasi PM2.5 TURUN %.4f ug/m3\n', abs(dC_dT));
    fprintf('  -> Suhu tinggi menyebabkan konveksi udara, polutan terbawa ke atas (menyebar)\n');
else
    fprintf('dC/dT > 0 : Ketika suhu naik 1 degC, konsentrasi PM2.5 NAIK %.4f ug/m3\n', dC_dT);
    fprintf('  -> Suhu tinggi meningkatkan reaksi kimia pembentukan polutan sekunder\n');
end

if dC_dH > 0
    fprintf('\ndC/dH > 0 : Ketika kelembaban naik 1%%, konsentrasi PM2.5 NAIK %.4f ug/m3\n', dC_dH);
    fprintf('  -> Kelembaban tinggi menyebabkan partikel PM2.5 menggumpal (hygroscopic growth)\n');
else
    fprintf('\ndC/dH < 0 : Ketika kelembaban naik 1%%, konsentrasi PM2.5 TURUN %.4f ug/m3\n', abs(dC_dH));
    fprintf('  -> Kelembaban tinggi menyebabkan partikel mengendap (wet deposition)\n');
end

if dC_dW < 0
    fprintf('\ndC/dW < 0 : Ketika angin naik 1 m/s, konsentrasi PM2.5 TURUN %.4f ug/m3\n', abs(dC_dW));
    fprintf('  -> Angin kencang mengencerkan dan menyebarkan polutan ke area lebih luas\n');
else
    fprintf('\ndC/dW > 0 : Ketika angin naik 1 m/s, konsentrasi PM2.5 NAIK %.4f ug/m3\n', dC_dW);
    fprintf('  -> Angin membawa polutan dari daerah lain ke titik pemantauan\n');
end

%% Bagian 6
% Analisis Paparan Polutan Menggunakan Integral
data_asli = A;
profil_harian = mean(data_asli, 1); % Menghitung nilai mean tiap kolom sensor sepanjang tahun

% Membentuk domain kontinu waktu (t) dari jam 0 hingga jam 24.
t_data = linspace(0, 24, length(profil_harian));

% Menentukan batas-batas integrasi batas bawah (a) dan batas atas (b)
t_pagi_mulai = 6; t_pagi_selesai = 14; % Jendela waktu Pagi - Siang (8 jam)
t_malam_mulai = 14; t_malam_selesai = 22; % Jendela waktu Siang - Malam (8 jam)

% Digunakan polinomial derajat 7 (orde 7) untuk memodelkan fluktuasi polutan udara secara smooth
orde_poly = 7;
p_integral = polyfit(t_data, profil_harian, orde_poly);

% Proses Integrasi Simbolik 
p_integrasi = [p_integral ./ (orde_poly+1:-1:1), 0];

% Menghitung akumulasi eksak untuk Periode 1 (Pagi - Siang: 06.00 s.d 14.00)
analitik_pagi = polyval(p_integrasi, t_pagi_selesai) - polyval(p_integrasi, t_pagi_mulai);

% Menghitung akumulasi eksak untuk Periode 2 (Siang - Malam: 14.00 s.d 22.00)
analitik_malam = polyval(p_integrasi, t_malam_selesai) - polyval(p_integrasi, t_malam_mulai);

% Menggunakan pemetaan kondisi logis untuk mencari indeks baris di dalam matriks
idx_6 = find(t_data >= t_pagi_mulai, 1, 'first');
idx_14 = find(t_data >= t_pagi_selesai, 1, 'first');
idx_22 = find(t_data >= t_malam_selesai, 1, 'first');

% A. Perhitungan Aturan Trapesium untuk Jendela Pagi-Siang
t_pagi = t_data(idx_6:idx_14);       % Mengiris partisi waktu internal pagi
y_pagi = profil_harian(idx_6:idx_14);% Mengiris nilai polutan koordinat Y
N_pagi = length(t_pagi) - 1;         % Jumlah sub-interval partisi trapesium
h_pagi = (t_pagi(end) - t_pagi(1)) / N_pagi; % Menghitung lebar langkah (step size h)
% Implementasi Aturan Trapesium Komposisi: I = (h/2) * [y0 + 2*sum(y_i) + yN]
numerik_pagi = (h_pagi / 2) * (y_pagi(1) + 2*sum(y_pagi(2:end-1)) + y_pagi(end));

% B. Perhitungan Aturan Trapesium untuk Jendela Siang-Malam
t_malam = t_data(idx_14:idx_22);       % Mengiris partisi waktu internal malam
y_malam = profil_harian(idx_14:idx_22);% Mengiris nilai polutan koordinat Y
N_malam = length(t_malam) - 1;         % Jumlah sub-interval partisi trapesium
h_malam = (t_malam(end) - t_malam(1)) / N_malam; % Menghitung lebar langkah (step size h)
% Implementasi Aturan Trapesium Komposisi
numerik_malam = (h_malam / 2) * (y_malam(1) + 2*sum(y_malam(2:end-1)) + y_malam(end));

% Menghitung Galat Absolut: Nilai Mutlak dari |Solusi Analitik - Solusi Numerik|
err_abs_pagi = abs(analitik_pagi - numerik_pagi);
% Menghitung Galat Relatif dalam bentuk presentase terhadap Solusi Analitik Eksak
err_rel_pagi = (err_abs_pagi / analitik_pagi) * 100;
% Mengulangi perhitungan galat absolut dan relatif untuk sesi Siang-Malam
err_abs_malam = abs(analitik_malam - numerik_malam);
err_rel_malam = (err_abs_malam / analitik_malam) * 100;

% Proses pencetakan data numerik akhir secara terstruktur 
fprintf('\nHASIL INTEGRAL PAPARAN POLUTAN:\n');
fprintf('----------------------------------------------------------------\n');
fprintf('Periode Pagi-Siang (06.00 - 14.00):\n');
fprintf('   - Solusi Analitik (Poly-7)  : %.4f\n', analitik_pagi);
fprintf('   - Solusi Numerik (Trapezoid): %.4f\n', numerik_pagi);
fprintf('   - Galat Absolut             : %.4e\n', err_abs_pagi);
fprintf('   - Galat Relatif             : %.4f%%\n', err_rel_pagi);
fprintf('-----------------------------------------------------------------\n');
fprintf('Periode Siang-Malam (14.00 - 22.00):\n');
fprintf('   - Solusi Analitik (Poly-7)  : %.4f\n', analitik_malam);
fprintf('   - Solusi Numerik (Trapezoid): %.4f\n', numerik_malam);
fprintf('   - Galat Absolut             : %.4e\n', err_abs_malam);
fprintf('   - Galat Relatif             : %.4f%%\n', err_rel_malam);
fprintf('-----------------------------------------------------------------\n');

% visualisasi grafik
figure('Name', 'Analisis Integral Paparan Polutan', 'NumberTitle', 'off');

% Menggambar kurva kontinu beresolusi tinggi (500 titik) menggunakan koefisien polinomial
t_mulus = linspace(0, 24, 500);
y_mulus = polyval(p_integral, t_mulus);

% Plotting data asli diskret beserta kurva matematikanya
plot(t_data, profil_harian, 'k.', 'MarkerSize', 8, 'DisplayName', 'Data Rata-rata Polutan'); hold on;
plot(t_mulus, y_mulus, 'k-', 'LineWidth', 1.5, 'DisplayName', 'Kurva Pendekatan f(t)');

% Representasi Geometris Integral Sesi Pagi-Siang: Mewarnai area di bawah kurva dengan warna Hijau (g)
t_fill_pagi = linspace(t_pagi_mulai, t_pagi_selesai, 100);
y_fill_pagi = polyval(p_integral, t_fill_pagi);
fill([t_fill_pagi, fliplr(t_fill_pagi)], [y_fill_pagi, zeros(1,100)], 'g', ...
    'FaceAlpha', 0.3, 'DisplayName', sprintf('Pagi-Siang (Num: %.2f)', numerik_pagi));

% Representasi Geometris Integral Sesi Siang-Malam: Mewarnai area di bawah kurva dengan warna Merah (r)
t_fill_malam = linspace(t_malam_mulai, t_malam_selesai, 100);
y_fill_malam = polyval(p_integral, t_fill_malam);
fill([t_fill_malam, fliplr(t_fill_malam)], [y_fill_malam, zeros(1,100)], 'r', ...
    'FaceAlpha', 0.3, 'DisplayName', sprintf('Siang-Malam (Num: %.2f)', numerik_malam));

% Konfigurasi pelabelan aksis dan plot grafik ilmiah
title('Perbandingan Total Paparan Polutan berdasarkan Interval Waktu');
xlabel('Jam (Waktu) dalam 24 Jam');
ylabel('Konsentrasi Polutan');
xlim([0 24]);
xticks(0:2:24); % Membuat penanda waktu berjarak kelipatan 2 jam
grid on;
legend('Location', 'best');

%% Bagian 7
% Analisis Akhir Hasil Pemodelan Kualitas Udara

fprintf('\nBAGIAN 7: Analisis Akhir dan Kesimpulan\n');

% Mengumpulkan beberapa indikator utama dari hasil sebelumnya
energi_95 = energi_kum(k_efektif) * 100;

mean_err_taylor3 = mean(err3);
mean_err_taylor5 = mean(err5);
mean_err_taylor7 = mean(err7);

fprintf('\nRINGKASAN HASIL:\n');
fprintf('------------------------------------------------------\n');
fprintf('Rank efektif SVD (95%% energi)      : %d\n', k_efektif);
fprintf('Energi kumulatif                   : %.2f%%\n', energi_95);
fprintf('------------------------------------------------------\n');
fprintf('Galat rata-rata Taylor orde 3      : %.6f\n', mean_err_taylor3);
fprintf('Galat rata-rata Taylor orde 5      : %.6f\n', mean_err_taylor5);
fprintf('Galat rata-rata Taylor orde 7      : %.6f\n', mean_err_taylor7);
fprintf('------------------------------------------------------\n');
fprintf('Paparan pagi-siang                 : %.4f\n', numerik_pagi);
fprintf('Paparan siang-malam                : %.4f\n', numerik_malam);
fprintf('------------------------------------------------------\n');

% Membandingkan galat Taylor
figure('Name','Bagian 7 - Ringkasan Analisis','NumberTitle','off');

subplot(2,2,1);
bar([mean_err_taylor3 mean_err_taylor5 mean_err_taylor7]);
set(gca,'XTickLabel',{'Orde 3','Orde 5','Orde 7'});
ylabel('Galat Rata-rata');
title('Perbandingan Galat Taylor');
grid on;

% Membandingkan error rekonstruksi SVD
subplot(2,2,2);
plot(k_values, errors*100,'o-','LineWidth',1.5);
xlabel('Nilai k');
ylabel('Relative Error (%)');
title('Error Rekonstruksi SVD');
grid on;

% Perbandingan paparan polutan
subplot(2,2,3);
bar([numerik_pagi numerik_malam]);
set(gca,'XTickLabel',{'06-14','14-22'});
ylabel('Total Paparan');
title('Perbandingan Paparan Polutan');
grid on;

% Energi kumulatif SVD
subplot(2,2,4);
plot(energi_kum*100,'LineWidth',1.5);
hold on;
plot([1 length(energi_kum)],[95 95],'r--');
xlabel('Komponen');
ylabel('Energi Kumulatif (%)');
title('Energi Kumulatif SVD');
grid on;

fprintf('\nKESIMPULAN SEMENTARA:\n');

if mean_err_taylor7 < mean_err_taylor5 && mean_err_taylor5 < mean_err_taylor3
    fprintf('- Orde Taylor yang lebih tinggi menghasilkan galat yang lebih kecil.\n');
end

fprintf('- Semakin besar nilai k, error rekonstruksi SVD semakin kecil.\n');
fprintf('- Sebagian besar informasi data telah direpresentasikan oleh %d komponen utama.\n', k_efektif);

if numerik_pagi > numerik_malam
    fprintf('- Paparan polutan lebih besar pada periode pagi-siang.\n');
else
    fprintf('- Paparan polutan lebih besar pada periode siang-malam.\n');
end
