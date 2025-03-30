import React, { useState } from 'react';
import { Card, Button, Accordion, Badge } from 'react-bootstrap';
import styles from './DebugPanel.module.css';

/**
 * 调试面板组件
 * 显示调试信息、状态和运行时错误
 */
const DebugPanel = ({ info = {}, errors = [], warnings = [], onClear }) => {
  const [expanded, setExpanded] = useState(false);
  const hasErrors = errors.length > 0;
  const hasWarnings = warnings.length > 0;

  const toggleExpand = () => {
    setExpanded(!expanded);
  };

  return (
    <div className={styles.debugPanel}>
      <Card className={`${styles.debugCard} ${hasErrors ? styles.hasErrors : ''}`}>
        <Card.Header
          className={styles.debugHeader}
          onClick={toggleExpand}
          style={{ cursor: 'pointer' }}
        >
          <div className={styles.headerContent}>
            <span>调试面板</span>
            {hasErrors && (
              <Badge bg="danger" className="ms-2">
                错误 ({errors.length})
              </Badge>
            )}
            {hasWarnings && (
              <Badge bg="warning" className="ms-2">
                警告 ({warnings.length})
              </Badge>
            )}
            <Button
              variant="link"
              size="sm"
              className={styles.toggleButton}
              onClick={e => {
                e.stopPropagation();
                toggleExpand();
              }}
            >
              {expanded ? '收起' : '展开'}
            </Button>
          </div>
        </Card.Header>

        {expanded && (
          <Card.Body className={styles.debugBody}>
            {/* 错误列表 */}
            {hasErrors && (
              <div className={styles.section}>
                <h6 className={styles.sectionTitle}>错误</h6>
                <ul className={styles.errorList}>
                  {errors.map((error, index) => (
                    <li key={`error-${index}`} className={styles.errorItem}>
                      {error.message || error.toString()}
                      {error.stack && (
                        <div className={styles.errorStack}>
                          <pre>{error.stack}</pre>
                        </div>
                      )}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* 警告列表 */}
            {hasWarnings && (
              <div className={styles.section}>
                <h6 className={styles.sectionTitle}>警告</h6>
                <ul className={styles.warningList}>
                  {warnings.map((warning, index) => (
                    <li key={`warning-${index}`} className={styles.warningItem}>
                      {warning}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* 状态信息 */}
            <div className={styles.section}>
              <h6 className={styles.sectionTitle}>状态信息</h6>
              <Accordion defaultActiveKey="0" className={styles.infoAccordion}>
                {Object.entries(info).map(([key, value], index) => (
                  <Accordion.Item eventKey={index.toString()} key={key}>
                    <Accordion.Header>{key}</Accordion.Header>
                    <Accordion.Body>
                      <pre className={styles.jsonDisplay}>
                        {typeof value === 'object' ? JSON.stringify(value, null, 2) : value}
                      </pre>
                    </Accordion.Body>
                  </Accordion.Item>
                ))}
              </Accordion>
            </div>

            {/* 清除按钮 */}
            {(hasErrors || hasWarnings) && onClear && (
              <div className={styles.buttonContainer}>
                <Button variant="outline-secondary" size="sm" onClick={onClear}>
                  清除所有消息
                </Button>
              </div>
            )}
          </Card.Body>
        )}
      </Card>
    </div>
  );
};

export default DebugPanel;
