# BÁO CÁO CẤU TRÚC VÀ CHẤT LƯỢNG DỮ LIỆU THÔ (DATA PROFILING REPORT)

Bản báo cáo này cung cấp thông số chi tiết về cấu trúc, tỷ lệ khuyết thiếu và đặc điểm phân bố của dữ liệu nguồn trước khi thực hiện tiền xử lý.

## Bảng: `customers.csv`
- Tổng số dòng: 121,930
- Tổng số cột: 7

| Tên Cột | Kiểu Dữ Liệu | Null (%) | Unique | Phân bố / Giá trị (Distribution) | Ghi chú (Flags) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `customer_id` | `int64` | 0.0% | 121,930 | Range: 1 -> 157563 | - |
| `zip` | `int64` | 0.0% | 31,491 | Range: 1001 -> 99950 | - |
| `city` | `str` | 0.0% | 42 | Top 5: Cam Pha, Thai Nguyen, Phu Ly, Hanoi, Ha Long... | - |
| `signup_date` | `str` | 0.0% | 3,941 | Top 5: 2022-06-02, 2022-11-13, 2022-11-15, 2021-10-31, 2022-07-18... | - |
| `gender` | `str` | 0.0% | 3 | Values: Female, Male, Non-binary | - |
| `age_group` | `str` | 0.0% | 5 | Values: 35-44, 45-54, 18-24, 55+, 25-34 | - |
| `acquisition_channel` | `str` | 0.0% | 6 | Values: social_media, email_campaign, organic_search, referral, direct, paid_search | - |

---

## Bảng: `geography.csv`
- Tổng số dòng: 39,948
- Tổng số cột: 4

| Tên Cột | Kiểu Dữ Liệu | Null (%) | Unique | Phân bố / Giá trị (Distribution) | Ghi chú (Flags) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `zip` | `int64` | 0.0% | 39,948 | Range: 1 -> 99950 | - |
| `city` | `str` | 0.0% | 42 | Top 5: Cam Pha, Phu Ly, Thai Nguyen, Hanoi, Nam Dinh... | - |
| `region` | `str` | 0.0% | 3 | Values: East, Central, West | - |
| `district` | `str` | 0.0% | 39 | Top 5: District #25, District #30, District #19, District #24, District #07... | - |

---

## Bảng: `inventory.csv`
- Tổng số dòng: 60,247
- Tổng số cột: 17

| Tên Cột | Kiểu Dữ Liệu | Null (%) | Unique | Phân bố / Giá trị (Distribution) | Ghi chú (Flags) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `snapshot_date` | `str` | 0.0% | 126 | Top 5: 2018-05-31, 2021-03-31, 2018-04-30, 2018-06-30, 2021-04-30... | - |
| `product_id` | `int64` | 0.0% | 1,624 | Range: 1 -> 2412 | - |
| `stock_on_hand` | `int64` | 0.0% | 1,895 | Range: 3 -> 2673 | - |
| `units_received` | `int64` | 0.0% | 360 | Range: 1 -> 817 | - |
| `units_sold` | `int64` | 0.0% | 303 | Range: 1 -> 670 | - |
| `stockout_days` | `int64` | 0.0% | 29 | Range: 0 -> 28 | - |
| `days_of_supply` | `float64` | 0.0% | 9,289 | Range: 5.2 -> 68100 | - |
| `fill_rate` | `float64` | 0.0% | 29 | Range: 0.0667 -> 1 | - |
| `stockout_flag` | `int64` | 0.0% | 2 | Range: 0 -> 1 | - |
| `overstock_flag` | `int64` | 0.0% | 2 | Range: 0 -> 1 | - |
| `reorder_flag` | `int64` | 0.0% | 1 | Range: 0 -> 0 | CONSTANT |
| `sell_through_rate` | `float64` | 0.0% | 4,017 | Range: 0.0004 -> 0.8531 | - |
| `product_name` | `str` | 0.0% | 1,465 | Top 5: VietMode RP-09, VietMode RP-10, VietMode RP-11, VietMode RP-12, VietMode RP-33... | - |
| `category` | `str` | 0.0% | 4 | Values: Casual, Outdoor, Streetwear, GenZ | - |
| `segment` | `str` | 0.0% | 8 | Values: All-weather, Activewear, Premium, Balanced, Standard, Performance, Everyday, Trendy | - |
| `year` | `int64` | 0.0% | 11 | Range: 2012 -> 2022 | - |
| `month` | `int64` | 0.0% | 12 | Range: 1 -> 12 | - |

---

## Bảng: `order_items.csv`
- Tổng số dòng: 714,669
- Tổng số cột: 7

| Tên Cột | Kiểu Dữ Liệu | Null (%) | Unique | Phân bố / Giá trị (Distribution) | Ghi chú (Flags) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `order_id` | `int64` | 0.0% | 646,945 | Range: 1 -> 834397 | - |
| `product_id` | `int64` | 0.0% | 1,598 | Range: 1 -> 2412 | - |
| `quantity` | `int64` | 0.0% | 8 | Range: 1 -> 8 | - |
| `unit_price` | `float64` | 0.0% | 501,330 | Range: 392.57 -> 43056 | - |
| `discount_amount` | `float64` | 0.0% | 204,449 | Range: 0 -> 35235.5 | - |
| `promo_id` | `str` | 61.3% | 50 | Top 5: PROMO-0014, PROMO-0010, PROMO-0004, PROMO-0020, PROMO-0011... | - |
| `promo_id_2` | `str` | 100.0% | 2 | Values: PROMO-0015, PROMO-0025 | HIGH NULL |

---

## Bảng: `orders.csv`
- Tổng số dòng: 646,945
- Tổng số cột: 8

| Tên Cột | Kiểu Dữ Liệu | Null (%) | Unique | Phân bố / Giá trị (Distribution) | Ghi chú (Flags) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `order_id` | `int64` | 0.0% | 646,945 | Range: 1 -> 834397 | - |
| `order_date` | `str` | 0.0% | 3,833 | Top 5: 2018-05-30, 2018-05-31, 2017-03-30, 2018-06-01, 2014-04-29... | - |
| `customer_id` | `int64` | 0.0% | 90,246 | Range: 1 -> 157563 | - |
| `zip` | `int64` | 0.0% | 29,932 | Range: 1001 -> 99950 | - |
| `order_status` | `str` | 0.0% | 6 | Values: delivered, returned, shipped, cancelled, paid, created | - |
| `payment_method` | `str` | 0.0% | 5 | Values: credit_card, cod, paypal, apple_pay, bank_transfer | - |
| `device_type` | `str` | 0.0% | 3 | Values: desktop, mobile, tablet | - |
| `order_source` | `str` | 0.0% | 6 | Values: paid_search, direct, referral, email_campaign, organic_search, social_media | - |

---

## Bảng: `payments.csv`
- Tổng số dòng: 646,945
- Tổng số cột: 4

| Tên Cột | Kiểu Dữ Liệu | Null (%) | Unique | Phân bố / Giá trị (Distribution) | Ghi chú (Flags) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `order_id` | `int64` | 0.0% | 646,945 | Range: 1 -> 834397 | - |
| `payment_method` | `str` | 0.0% | 5 | Values: credit_card, cod, paypal, apple_pay, bank_transfer | - |
| `payment_value` | `float64` | 0.0% | 595,420 | Range: 389.74 -> 331570 | - |
| `installments` | `int64` | 0.0% | 5 | Range: 1 -> 12 | - |

---

## Bảng: `products.csv`
- Tổng số dòng: 2,412
- Tổng số cột: 8

| Tên Cột | Kiểu Dữ Liệu | Null (%) | Unique | Phân bố / Giá trị (Distribution) | Ghi chú (Flags) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `product_id` | `int64` | 0.0% | 2,412 | Range: 1 -> 2412 | - |
| `product_name` | `str` | 0.0% | 2,172 | Top 5: VietMode RP-01, VietMode RP-02, VietMode RP-03, VietMode RP-04, VietMode RP-05... | - |
| `category` | `str` | 0.0% | 4 | Values: Streetwear, Casual, Outdoor, GenZ | - |
| `segment` | `str` | 0.0% | 8 | Values: Everyday, Performance, Balanced, Standard, All-weather, Premium, Trendy, Activewear | - |
| `size` | `str` | 0.0% | 4 | Values: S, M, L, XL | - |
| `color` | `str` | 0.0% | 10 | Values: green, silver, pink, yellow, red, black, orange, blue, white, purple | - |
| `price` | `float64` | 0.0% | 1,990 | Range: 9.05659 -> 40950 | - |
| `cogs` | `float64` | 0.0% | 2,381 | Range: 5.18383 -> 38902.5 | - |

---

## Bảng: `promotions.csv`
- Tổng số dòng: 50
- Tổng số cột: 10

| Tên Cột | Kiểu Dữ Liệu | Null (%) | Unique | Phân bố / Giá trị (Distribution) | Ghi chú (Flags) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `promo_id` | `str` | 0.0% | 50 | Top 5: PROMO-0001, PROMO-0002, PROMO-0003, PROMO-0004, PROMO-0005... | - |
| `promo_name` | `str` | 0.0% | 50 | Top 5: Spring Sale 2013, Mid-Year Sale 2013, Fall Launch 2013, Year-End Sale 2013, Urban Blowout 2013... | - |
| `promo_type` | `str` | 0.0% | 2 | Values: percentage, fixed | - |
| `discount_value` | `float64` | 0.0% | 6 | Range: 10 -> 50 | - |
| `start_date` | `str` | 0.0% | 50 | Top 5: 2013-03-18, 2013-06-23, 2013-08-30, 2013-11-18, 2013-07-30... | - |
| `end_date` | `str` | 0.0% | 50 | Top 5: 2013-04-17, 2013-07-22, 2013-10-02, 2014-01-02, 2013-09-02... | - |
| `applicable_category` | `str` | 80.0% | 2 | Values: Streetwear, Outdoor | - |
| `promo_channel` | `str` | 0.0% | 5 | Values: email, online, all_channels, in_store, social_media | - |
| `stackable_flag` | `int64` | 0.0% | 2 | Range: 0 -> 1 | - |
| `min_order_value` | `int64` | 0.0% | 5 | Range: 0 -> 200000 | - |

---

## Bảng: `returns.csv`
- Tổng số dòng: 39,939
- Tổng số cột: 7

| Tên Cột | Kiểu Dữ Liệu | Null (%) | Unique | Phân bố / Giá trị (Distribution) | Ghi chú (Flags) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `return_id` | `str` | 0.0% | 39,939 | Top 5: RET-000001, RET-000002, RET-000003, RET-000004, RET-000005... | - |
| `order_id` | `int64` | 0.0% | 36,062 | Range: 2 -> 833351 | - |
| `product_id` | `int64` | 0.0% | 1,286 | Range: 3 -> 2412 | - |
| `return_date` | `str` | 0.0% | 3,806 | Top 5: 2016-06-18, 2013-06-19, 2014-04-17, 2016-04-19, 2018-06-15... | - |
| `return_reason` | `str` | 0.0% | 5 | Values: late_delivery, wrong_size, defective, changed_mind, not_as_described | - |
| `return_quantity` | `int64` | 0.0% | 8 | Range: 1 -> 8 | - |
| `refund_amount` | `float64` | 0.0% | 39,560 | Range: 458.81 -> 160938 | - |

---

## Bảng: `reviews.csv`
- Tổng số dòng: 113,551
- Tổng số cột: 7

| Tên Cột | Kiểu Dữ Liệu | Null (%) | Unique | Phân bố / Giá trị (Distribution) | Ghi chú (Flags) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `review_id` | `str` | 0.0% | 113,551 | Top 5: REV-0000001, REV-0000002, REV-0000003, REV-0000005, REV-0000006... | - |
| `order_id` | `int64` | 0.0% | 111,369 | Range: 1 -> 833296 | - |
| `product_id` | `int64` | 0.0% | 1,412 | Range: 3 -> 2412 | - |
| `customer_id` | `int64` | 0.0% | 48,676 | Range: 2 -> 157563 | - |
| `review_date` | `str` | 0.0% | 3,825 | Top 5: 2017-05-06, 2014-05-11, 2017-06-14, 2017-06-19, 2015-05-10... | - |
| `rating` | `int64` | 0.0% | 5 | Range: 1 -> 5 | - |
| `review_title` | `str` | 0.0% | 18 | Top 5: Very satisfied, Highly recommend, Great quality, Excellent product!, Good overall... | - |

---

## Bảng: `sales.csv`
- Tổng số dòng: 3,833
- Tổng số cột: 3

| Tên Cột | Kiểu Dữ Liệu | Null (%) | Unique | Phân bố / Giá trị (Distribution) | Ghi chú (Flags) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Date` | `str` | 0.0% | 3,833 | Top 5: 2012-07-04, 2012-07-05, 2012-07-06, 2012-07-07, 2012-07-08... | - |
| `Revenue` | `float64` | 0.0% | 3,833 | Range: 279814 -> 2.09053e+07 | - |
| `COGS` | `float64` | 0.0% | 3,833 | Range: 236576 -> 1.65359e+07 | - |

---

## Bảng: `shipments.csv`
- Tổng số dòng: 566,067
- Tổng số cột: 4

| Tên Cột | Kiểu Dữ Liệu | Null (%) | Unique | Phân bố / Giá trị (Distribution) | Ghi chú (Flags) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `order_id` | `int64` | 0.0% | 566,067 | Range: 1 -> 834325 | - |
| `ship_date` | `str` | 0.0% | 3,831 | Top 5: 2018-06-02, 2017-06-03, 2018-06-03, 2018-06-01, 2017-06-02... | - |
| `delivery_date` | `str` | 0.0% | 3,831 | Top 5: 2018-06-06, 2018-06-07, 2018-06-08, 2016-05-05, 2016-05-04... | - |
| `shipping_fee` | `float64` | 0.0% | 1,856 | Range: 0 -> 32 | - |

---

## Bảng: `web_traffic.csv`
- Tổng số dòng: 3,652
- Tổng số cột: 7

| Tên Cột | Kiểu Dữ Liệu | Null (%) | Unique | Phân bố / Giá trị (Distribution) | Ghi chú (Flags) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `date` | `str` | 0.0% | 3,652 | Top 5: 2013-01-01, 2013-01-02, 2013-01-03, 2013-01-04, 2013-01-05... | - |
| `sessions` | `int64` | 0.0% | 3,447 | Range: 7973 -> 50947 | - |
| `unique_visitors` | `int64` | 0.0% | 3,382 | Range: 6136 -> 40430 | - |
| `page_views` | `int64` | 0.0% | 3,620 | Range: 30451 -> 275560 | - |
| `bounce_rate` | `float64` | 0.0% | 261 | Range: 0.0032 -> 0.0058 | - |
| `avg_session_duration_sec` | `float64` | 0.0% | 1,771 | Range: 100.1 -> 319.9 | - |
| `traffic_source` | `str` | 0.0% | 6 | Values: organic_search, direct, referral, social_media, paid_search, email_campaign | - |

---

## Phụ lục: Giải thích các Ghi chú (Flags)
- **CONSTANT:** Cột chỉ chứa duy nhất một giá trị (biến thiên bằng 0). Không mang lại giá trị phân tích, cần cân nhắc loại bỏ.
- **HIGH NULL:** Cột có tỷ lệ dữ liệu trống từ 90% trở lên. Cần kiểm tra lại quy trình thu thập hoặc nghiệp vụ liên quan.
