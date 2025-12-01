package com.example.capstone_design.dto;

import com.example.capstone_design.entity.Recording;
import lombok.AllArgsConstructor;
import lombok.Getter;

import java.time.LocalDateTime;

@Getter
@AllArgsConstructor
public class RecordingResponse {
    private Long id;
    private String uploader;
    private String originalFilename;
    private String storedFilename;
    private String contentType;
    private long size;
    private LocalDateTime createdAt;

    // ===== 분석 결과 =====
    private String emotion;
    private Double confidence;

    // ===== 위치 정보 =====
    private Double latitude;
    private Double longitude;
    private String address;

    // ===== 사용자 입력 =====
    private String description;
    private boolean emotionPublic;

    public static RecordingResponse from(Recording r) {
        return new RecordingResponse(
                r.getId(),
                r.getUploader(),
                r.getOriginalFilename(),
                r.getStoredFilename(),
                r.getContentType(),
                r.getSize(),
                r.getCreatedAt(),
                r.getEmotion(),
                r.getConfidence(),
                r.getLatitude(),
                r.getLongitude(),
                r.getAddress(),
                r.getDescription(),
                r.isEmotionPublic()
        );
    }
}
