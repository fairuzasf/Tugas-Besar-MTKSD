clear; clc;

% Membaca dataset dari file CSV
data = readtable('DataSet_KualitasUdara_K1.csv', ...
    'VariableNamingRule', 'preserve');
% Mengekstrak  data (kolom 2 sampai 97)
A = table2array(data(:, 2:end));

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



%% Bagian 4

% Mengambil data waktu dan polutan di satu titik (titik ke-1 misal)
x_data = data{:, 1};       % Hari ke-
y_data = data{:, 2};       % PM2.5 di titik 1

% Mencari titik puncak polusi (maksimum)
[max_val, max_idx] = max(y_data);
x0 = x_data(max_idx);      % Hari saat puncak terjadi

fprintf('Titik puncak polusi terjadi pada hari ke-%d dengan PM2.5 = %.2f\n', x0, max_val);

% Membuat window (jendela data) di sekitar titik puncak
% Mengambil contoh 10 hari sebelum dan 10 hari sesudah puncak
win_size = 10;
idx_start = max(1, max_idx - win_size);
idx_end = min(length(x_data), max_idx + win_size);

x_win = x_data(idx_start:idx_end);
y_win = y_data(idx_start:idx_end);

% Membuat Fungsi Kontinu f(x) menggunakan Polinomial
% Kita fit dengan polinomial orde 8 agar kurvanya mulus dan bisa diturunkan sampai orde 7
p = polyfit(x_win, y_win, 8); 

% Menghitung Turunan Analitik dari Polinomial p(x)
p1 = polyder(p);   % Turunan ke-1
p2 = polyder(p1);  % Turunan ke-2
p3 = polyder(p2);  % Turunan ke-3
p4 = polyder(p3);  % Turunan ke-4
p5 = polyder(p4);  % Turunan ke-5
p6 = polyder(p5);  % Turunan ke-6
p7 = polyder(p6);  % Turunan ke-7

% Mengevaluasi nilai turunan tepat di titik puncak (x0)
d0 = polyval(p, x0);
d1 = polyval(p1, x0);
d2 = polyval(p2, x0);
d3 = polyval(p3, x0);
d4 = polyval(p4, x0);
d5 = polyval(p5, x0);
d6 = polyval(p6, x0);
d7 = polyval(p7, x0);

% Membangun Deret Taylor Orde 3, 5, dan 7
dx = x_win - x0; % (x - x0)

% Taylor Orde 3
T3 = d0 + d1.*dx + (d2/factorial(2)).*dx.^2 + (d3/factorial(3)).*dx.^3;

% Taylor Orde 5 (Melanjutkan dari Taylor 3)
T5 = T3 + (d4/factorial(4)).*dx.^4 + (d5/factorial(5)).*dx.^5;

% Taylor Orde 7 (Melanjutkan dari Taylor 5)
T7 = T5 + (d6/factorial(6)).*dx.^6 + (d7/factorial(7)).*dx.^7;

% 8. Menghitung Galat Aproksimasi Analitik (Absolute Error)
% Galat dihitung terhadap fungsi kontinu f(x) yang kita buat
f_true = polyval(p, x_win);
err3 = abs(f_true - T3);
err5 = abs(f_true - T5);
err7 = abs(f_true - T7);

% Visualisasi nihh

% Grafik pertamaa: Perbandingan Fungsi dan Deret Taylor
figure;
plot(x_win, y_win, 'ko', 'DisplayName', 'Data Observasi Asli'); hold on;
plot(x_win, f_true, 'k-', 'LineWidth', 2, 'DisplayName', 'Fungsi Hampiran f(x)');
plot(x_win, T3, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Taylor Orde 3');
plot(x_win, T5, 'g-.', 'LineWidth', 1.5, 'DisplayName', 'Taylor Orde 5');
plot(x_win, T7, 'b:', 'LineWidth', 1.5, 'DisplayName', 'Taylor Orde 7');

% Menandai titik puncaknyaa
plot(x0, max_val, 'm*', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'Titik Puncak (x_0)');

legend('Location', 'best');
title('Aproksimasi Deret Taylor di Sekitar Titik Puncak PM2.5');
xlabel('Hari Ke-'); 
ylabel('Konsentrasi PM2.5');
grid on;

% Grafik keduaaa: Perbandingan Galat
figure;
plot(x_win, err3, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Galat Orde 3'); hold on;
plot(x_win, err5, 'g-', 'LineWidth', 1.5, 'DisplayName', 'Galat Orde 5');
plot(x_win, err7, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Galat Orde 7');

legend('Location', 'best');
title('Perbandingan Galat Analitik Deret Taylor');
xlabel('Hari Ke-'); 
ylabel('Galat Absolut (|f(x) - T_n(x)|)');
grid on;

%% Bagian 5

%% Bagian 6

%% Bagian 7
