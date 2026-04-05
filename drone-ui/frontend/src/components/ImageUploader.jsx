import React, { useCallback, useState } from 'react';
import { UploadCloud, Image as ImageIcon } from 'lucide-react';

const ImageUploader = ({ onImageSelect, selectedImagePreview }) => {
  const [isDragActive, setIsDragActive] = useState(false);

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setIsDragActive(true);
    } else if (e.type === 'dragleave') {
      setIsDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onImageSelect(e.dataTransfer.files[0]);
    }
  }, [onImageSelect]);

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      onImageSelect(e.target.files[0]);
    }
  };

  if (selectedImagePreview) {
    return (
      <div className="image-preview-container">
        <img src={selectedImagePreview} alt="Selected" className="image-preview" />
        <button 
          className="btn-secondary" 
          onClick={() => onImageSelect(null)}
        >
          <UploadCloud size={20} />
          Choose Another Image
        </button>
      </div>
    );
  }

  return (
    <label 
      className={`glass-panel uploader-container ${isDragActive ? 'drag-active' : ''}`}
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
    >
      <input 
        type="file" 
        className="uploader-input" 
        accept="image/*" 
        onChange={handleChange} 
      />
      <ImageIcon className="uploader-icon" />
      <h3 style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>Upload Target Image</h3>
      <p style={{ color: 'var(--text-secondary)' }}>Drag & drop or click to browse</p>
    </label>
  );
};

export default ImageUploader;
